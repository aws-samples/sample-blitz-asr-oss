"""
Refresh AWS EC2 pricing references for the Blitz GPU price fields.

Queries the AWS Pricing API for the configured instance type/region and prints
a diff-ready snippet for the four ``gpu_price_*`` defaults in
``blitz/config.py`` and the matching ``GPU_PRICE_*`` keys in ``.env.example``.

This script is intentionally read-only. Review the output and paste the numbers
into the source files manually — auto-editing defaults from a remote API is a
supply-chain risk.

Spot rates are real-time and sampled from ``DescribeSpotPriceHistory`` across
all availability zones in the region over the last 7 days; the default stored
in config is the upper end of that range.

Requirements:
    pip install boto3
    AWS credentials with ``pricing:GetProducts`` and ``ec2:DescribeSpotPriceHistory``.

Usage:
    python scripts/update_pricing.py
    python scripts/update_pricing.py --instance-type g6.xlarge --region us-east-1
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, NamedTuple, Optional


class RiKey(NamedTuple):
    lease: str  # "1yr" | "3yr"
    offering_class: str  # "standard" | "convertible"
    purchase_option: str  # "No Upfront" | "Partial Upfront" | "All Upfront"

logger = logging.getLogger(__name__)

# AWS Pricing API is only served from us-east-1 and ap-south-1 regardless of
# which region you are pricing.
_PRICING_ENDPOINT_REGION = "us-east-1"


def _flatten_reserved(terms: dict) -> Dict[RiKey, Dict[str, float]]:
    """Collect Reserved Instance rates keyed by :class:`RiKey`."""
    out: Dict[RiKey, Dict[str, float]] = {}
    for offer in terms.get("Reserved", {}).values():
        ta = offer["termAttributes"]
        key = RiKey(ta["LeaseContractLength"], ta["OfferingClass"], ta["PurchaseOption"])
        hourly = 0.0
        upfront = 0.0
        for dim in offer["priceDimensions"].values():
            price = float(dim["pricePerUnit"]["USD"])
            if dim["unit"] == "Hrs":
                hourly = price
            elif dim["unit"] == "Quantity":
                upfront = price
        effective = hourly
        if upfront:
            # Amortize the upfront fee over the term to get an effective hourly.
            years = int(ta["LeaseContractLength"].rstrip("yr"))
            effective = hourly + (upfront / (years * 365 * 24))
        out[key] = {"hourly": hourly, "upfront": upfront, "effective": effective}
    return out


def _get_ondemand_and_ri(instance_type: str, region: str) -> dict:
    import boto3

    client = boto3.client("pricing", region_name=_PRICING_ENDPOINT_REGION)
    filters = [
        {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
        {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
        {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
        {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
        {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
        {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
    ]
    response = client.get_products(
        ServiceCode="AmazonEC2", Filters=filters, MaxResults=1
    )
    if not response.get("PriceList"):
        raise RuntimeError(
            f"No pricing for {instance_type} in {region}. Check the instance type."
        )

    product = json.loads(response["PriceList"][0])
    on_demand = 0.0
    for offer in product["terms"]["OnDemand"].values():
        for dim in offer["priceDimensions"].values():
            on_demand = float(dim["pricePerUnit"]["USD"])
    return {
        "publication_date": product.get("publicationDate"),
        "on_demand": on_demand,
        "reserved": _flatten_reserved(product["terms"]),
    }


def _get_spot_range(instance_type: str, region: str) -> Optional[Dict[str, float]]:
    import boto3

    client = boto3.client("ec2", region_name=region)
    since = datetime.now(timezone.utc) - timedelta(days=7)
    paginator = client.get_paginator("describe_spot_price_history")
    prices: list[float] = []
    for page in paginator.paginate(
        InstanceTypes=[instance_type],
        ProductDescriptions=["Linux/UNIX"],
        StartTime=since,
    ):
        prices.extend(float(p["SpotPrice"]) for p in page["SpotPriceHistory"])
    if not prices:
        return None
    return {
        "min": min(prices),
        "median": statistics.median(prices),
        "max": max(prices),
        "samples": len(prices),
    }


_DEFAULT_INSTANCE_TYPE = "g6.xlarge"
_DEFAULT_REGION = "us-east-1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-type", default=_DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--region", default=_DEFAULT_REGION)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        pricing = _get_ondemand_and_ri(args.instance_type, args.region)
    except Exception as exc:
        logger.error("Failed to fetch on-demand/RI pricing: %s", exc)
        return 1

    try:
        spot = _get_spot_range(args.instance_type, args.region)
    except Exception as exc:
        logger.warning("Failed to fetch spot history: %s", exc)
        spot = None

    ri = pricing["reserved"]
    std_1yr_nu = ri.get(RiKey("1yr", "standard", "No Upfront"), {}).get("hourly", 0.0)
    std_3yr_nu = ri.get(RiKey("3yr", "standard", "No Upfront"), {}).get("hourly", 0.0)
    std_1yr_au = ri.get(RiKey("1yr", "standard", "All Upfront"), {}).get("effective", 0.0)
    std_3yr_au = ri.get(RiKey("3yr", "standard", "All Upfront"), {}).get("effective", 0.0)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(
        f"\nAWS EC2 {args.instance_type} ({args.region}) pricing "
        f"(Linux/Shared, observed {today})\n"
    )
    print(f"  On-Demand:                     ${pricing['on_demand']:.4f}/hr")
    print(f"  1-yr Standard RI, No Upfront:  ${std_1yr_nu:.4f}/hr")
    print(f"  1-yr Standard RI, All Upfront: ${std_1yr_au:.4f}/hr effective")
    print(f"  3-yr Standard RI, No Upfront:  ${std_3yr_nu:.4f}/hr")
    print(f"  3-yr Standard RI, All Upfront: ${std_3yr_au:.4f}/hr effective")
    if spot:
        print(
            f"  Spot (7-day range, {spot['samples']} samples): "
            f"${spot['min']:.4f} – ${spot['max']:.4f}/hr "
            f"(median ${spot['median']:.4f})"
        )
    else:
        print("  Spot: unavailable")

    spot_default = spot["max"] if spot else 0.0

    print("\nPaste into blitz/config.py (No Upfront, most common baseline;")
    print("gpu_price_spot uses the upper end of the 7-day range):")
    print(f"    gpu_price_on_demand    default={pricing['on_demand']:.4f}")
    print(f"    gpu_price_spot         default={spot_default:.4f}")
    print(f"    gpu_price_savings_1yr  default={std_1yr_nu:.4f}")
    print(f"    gpu_price_savings_3yr  default={std_3yr_nu:.4f}")
    print(f"Update the 'Last refreshed' comment in blitz/config.py to {today}.")

    print("\nPaste into .env.example:")
    print(f"    GPU_PRICE_ON_DEMAND={pricing['on_demand']:.4f}")
    print(f"    GPU_PRICE_SPOT={spot_default:.4f}")
    print(f"    GPU_PRICE_SAVINGS_1YR={std_1yr_nu:.4f}")
    print(f"    GPU_PRICE_SAVINGS_3YR={std_3yr_nu:.4f}")

    if args.instance_type != _DEFAULT_INSTANCE_TYPE or args.region != _DEFAULT_REGION:
        print(
            f"\nWARNING: --instance-type/--region differ from the shipped defaults "
            f"({_DEFAULT_INSTANCE_TYPE}/{_DEFAULT_REGION}). Also update "
            f"`instance_type` and `instance_region` in blitz/config.py and the "
            f"INSTANCE_TYPE / INSTANCE_REGION values in .env.example "
            f"(current run used {args.instance_type!r}/{args.region!r})."
        )

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

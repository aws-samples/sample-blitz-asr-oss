"""Shared cost/metrics math used by the HTTP and AWS Batch code paths.

The two paths have different response shapes (the HTTP ``/transcribe`` endpoint
and the S3 batch JSON writer each have independent consumers), so this module
only exposes the raw numbers — formatting and key naming stay with the caller.
"""

from __future__ import annotations

from typing import Dict, NamedTuple


class CostMetrics(NamedTuple):
    """Raw processing metrics and per-1000-min cost estimates.

    ``cost_per_1000_min`` keys mirror whatever keys were passed in ``pricing``
    (typically ``on_demand``, ``spot``, ``savings_1yr``, ``savings_3yr``).
    """

    rtf: float
    cost_per_1000_min: Dict[str, float]
    lang_detect_pct: float
    transcribe_pct: float


def compute_cost_metrics(
    total_processing_time: float,
    audio_duration_sec: float,
    lang_detect_time_ms: float,
    transcribe_time_seconds: float,
    pricing: Dict[str, float],
) -> CostMetrics:
    """Compute RTF, extrapolated per-1000-min costs, and a time breakdown."""
    rtf = (
        total_processing_time / audio_duration_sec
        if audio_duration_sec > 0
        else 0.0
    )
    processing_hours_for_1000_min = (60000 * rtf) / 3600

    cost_per_1000_min = {
        option: processing_hours_for_1000_min * rate
        for option, rate in pricing.items()
    }

    if total_processing_time > 0:
        lang_detect_pct = (lang_detect_time_ms / 1000) / total_processing_time * 100
        transcribe_pct = transcribe_time_seconds / total_processing_time * 100
    else:
        lang_detect_pct = 0.0
        transcribe_pct = 0.0

    return CostMetrics(
        rtf=rtf,
        cost_per_1000_min=cost_per_1000_min,
        lang_detect_pct=lang_detect_pct,
        transcribe_pct=transcribe_pct,
    )

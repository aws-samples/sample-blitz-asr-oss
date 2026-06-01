"""
S3 handler for AWS Batch processing.

Provides utilities for downloading audio files from S3 and uploading
transcription results.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3Handler:
    """
    S3 operations for batch transcription processing.

    Handles downloading audio files from S3 and uploading transcription
    results in JSON format.
    """

    def __init__(self, region: Optional[str] = None):
        """
        Initialize S3 handler.

        Args:
            region: AWS region. If None, uses AWS_REGION env var or default.
        """
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.s3_client = boto3.client("s3", region_name=self.region)
        logger.info(f"S3Handler initialized for region: {self.region}")

    def download_audio(
        self, bucket: str, key: str, local_dir: Optional[str] = None
    ) -> str:
        """
        Download audio file from S3 to local path.

        Args:
            bucket: S3 bucket name
            key: S3 object key (path to audio file)
            local_dir: Local directory to save file. If None, uses temp dir.

        Returns:
            Local path to downloaded file

        Raises:
            ClientError: If download fails
        """
        # Determine local path
        filename = Path(key).name
        if local_dir:
            local_path = os.path.join(local_dir, filename)
        else:
            local_path = os.path.join(tempfile.gettempdir(), filename)

        logger.info(f"Downloading s3://{bucket}/{key} to {local_path}")

        try:
            self.s3_client.download_file(bucket, key, local_path)
            file_size = os.path.getsize(local_path)
            logger.info(f"Downloaded {file_size:,} bytes to {local_path}")
            return local_path
        except ClientError as e:
            logger.error(f"Failed to download s3://{bucket}/{key}: {e}")
            raise

    def upload_json(self, data: dict, bucket: str, key: str) -> str:
        """
        Upload JSON data to S3.

        Args:
            data: Dictionary to serialize and upload
            bucket: S3 bucket name
            key: S3 object key (path for result)

        Returns:
            S3 URI of uploaded file

        Raises:
            ClientError: If upload fails
        """
        logger.info(f"Uploading JSON to s3://{bucket}/{key}")

        try:
            json_body = json.dumps(data, indent=2, ensure_ascii=False)
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json_body.encode("utf-8"),
                ContentType="application/json",
            )
            s3_uri = f"s3://{bucket}/{key}"
            logger.info(f"Uploaded JSON ({len(json_body):,} bytes) to {s3_uri}")
            return s3_uri
        except ClientError as e:
            logger.error(f"Failed to upload JSON to s3://{bucket}/{key}: {e}")
            raise

    def upload_file(
        self, local_path: str, bucket: str, key: str, content_type: Optional[str] = None
    ) -> str:
        """
        Upload local file to S3.

        Args:
            local_path: Path to local file
            bucket: S3 bucket name
            key: S3 object key
            content_type: MIME type (auto-detected if None)

        Returns:
            S3 URI of uploaded file

        Raises:
            ClientError: If upload fails
        """
        logger.info(f"Uploading {local_path} to s3://{bucket}/{key}")

        try:
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type

            self.s3_client.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
            s3_uri = f"s3://{bucket}/{key}"
            logger.info(f"Uploaded file to {s3_uri}")
            return s3_uri
        except ClientError as e:
            logger.error(f"Failed to upload {local_path} to s3://{bucket}/{key}: {e}")
            raise

    def get_object_metadata(self, bucket: str, key: str) -> dict:
        """
        Get metadata for an S3 object.

        Args:
            bucket: S3 bucket name
            key: S3 object key

        Returns:
            Dictionary with object metadata including ContentLength, ContentType, etc.

        Raises:
            ClientError: If object doesn't exist or access denied
        """
        try:
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            return {
                "content_length": response.get("ContentLength", 0),
                "content_type": response.get("ContentType", ""),
                "last_modified": response.get("LastModified"),
                "etag": response.get("ETag", "").strip('"'),
            }
        except ClientError as e:
            logger.error(f"Failed to get metadata for s3://{bucket}/{key}: {e}")
            raise

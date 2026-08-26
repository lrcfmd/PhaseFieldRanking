import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config


@dataclass
class S3Config:
    """Configuration for S3 client."""

    endpoint: str
    access_key: str
    secret_key: str
    bucket: str

    @classmethod
    def from_env(cls) -> "S3Config":
        """Create S3Config from environment variables."""
        return cls(
            endpoint=os.environ.get("S3_ENDPOINT", "http://minio:9000"),
            access_key=os.environ.get("S3_ACCESS_KEY", "minioadmin"),
            secret_key=os.environ.get("S3_SECRET_KEY", "minioadmin"),
            bucket=os.environ.get("S3_BUCKET", "lmds"),
        )


class S3Client:
    """S3 client for downloading and uploading files."""

    def __init__(self, config: S3Config):
        self.config = config
        self.client = boto3.client(
            "s3",
            endpoint_url=config.endpoint,
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )

    def download(self, uri: str, dest: Path) -> None:
        """Download a file from S3 URI to local path."""
        parsed = urlparse(uri)
        if parsed.scheme != "s3":
            raise ValueError(f"Expected s3:// URI, got: {uri}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(parsed.netloc, parsed.path.lstrip("/"), str(dest))

    def upload(self, file_path: Path, key: str) -> None:
        """Upload a local file to S3."""
        self.client.upload_file(str(file_path), self.config.bucket, key)

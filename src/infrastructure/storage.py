from pathlib import Path
from urllib.parse import urlparse

from infrastructure.s3 import S3Client


class StorageService:
    """Service for handling file storage operations (downloads)."""

    def __init__(self, s3_client: S3Client, logger):
        self.s3 = s3_client
        self.logger = logger

    def download_inputs(
        self, inputs: dict, input_dir: Path
    ) -> dict[str, tuple[Path, str]]:
        """Download URI inputs to local files.

        Args:
            inputs: Dict of field_name -> field_spec where field_spec has "uri" and "mime_type"
            input_dir: Directory to download files to

        Returns:
            Dict of field_name -> (local_path, mime_type)
        """
        downloaded = {}
        input_dir.mkdir(parents=True, exist_ok=True)

        for field_name, field_spec in inputs.items():
            if isinstance(field_spec, dict) and "uri" in field_spec:
                uri = field_spec["uri"]
                mime_type = field_spec.get("mime_type", "application/octet-stream")

                parsed = urlparse(uri)
                filename = Path(parsed.path).name or f"{field_name}.bin"
                local_path = input_dir / filename

                self.s3.download(uri, local_path)
                downloaded[field_name] = (local_path, mime_type)
                self.logger.debug("downloaded", uri=uri, path=str(local_path))

        return downloaded

    def download_to_path(self, uri: str, dest: Path) -> None:
        """Download a file from an S3 URI to an exact local path."""
        parsed = urlparse(uri)
        if parsed.scheme != "s3":
            raise ValueError(f"Expected s3:// URI, got: {uri}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download(uri, dest)
        self.logger.debug("downloaded to path", uri=uri, path=str(dest))

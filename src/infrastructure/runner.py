import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import structlog

from infrastructure.logging import newLogger
from infrastructure.s3 import S3Client, S3Config
from infrastructure.storage import StorageService
from infrastructure.handler import ModelHandler


def _resolve_data_dir() -> Path:
    """Resolve data directory for local runs.

    In a container, DATA_DIR is explicitly set by the orchestrator.
    When running from source, derive from this file's location so the
    user doesn't have to be in the template root.
    """
    if env := os.environ.get("DATA_DIR"):
        return Path(env)
    module_dir = Path(__file__).resolve().parent
    project_data = module_dir.parent.parent / "data"
    if project_data.exists():
        return project_data
    return Path.cwd() / "data"


DATA_DIR = _resolve_data_dir()
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"


class ModelRunner:
    """Orchestrates model execution, Mode-driven and storage-agnostic."""

    def __init__(
        self,
        handler: ModelHandler,
        local_mode: bool = False,
        config: S3Config | None = None,
    ):
        self.handler = handler
        self.local_mode = local_mode
        self.config = config
        if self.local_mode:
            self.storage = None
            self.config = None
            self.runner_logger = newLogger("infrastructure.runner")
        else:
            assert config is not None, "S3Config is required when not in local mode"
            self.storage_logger = newLogger("services.storage")
            self.storage = StorageService(S3Client(config), self.storage_logger)
            self.runner_logger = newLogger("infrastructure.runner")

    def run(self, args: list[str] | None = None) -> None:
        """Run the model."""
        parsed = self._parse_args(args)
        try:
            exec_config = json.loads(parsed.parameters)
        except json.JSONDecodeError as e:
            self.runner_logger.error("Invalid --parameters JSON", error=str(e))
            sys.exit(1)

        run_id = exec_config.get("run_id", "unknown")
        mode = exec_config["mode"]
        inputs = exec_config.get("inputs", {})
        parameters = exec_config.get("parameters", {})
        output_path = exec_config.get("output", {}).get("path", "")

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(run_id=run_id, mode=mode)

        self.runner_logger.info("Starting model processing")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Download runtime assets before resolving inputs so the handler can
        # read weights, lookup tables, etc. from their declared paths.
        assets = exec_config.get("assets", [])
        if assets and not self.local_mode and self.storage is not None:
            for asset in assets:
                target_path = Path(asset["path"])
                self.storage.download_to_path(asset["uri"], target_path)
            self.runner_logger.info("Downloaded runtime assets", count=len(assets))

        # Route each input field to a value or a file by its wire shape — not by
        # the Mode name. A Mode may freely mix value and file inputs.
        values, files = self._resolve_inputs(inputs)
        self.runner_logger.info(
            "Resolved inputs", values=len(values), files=len(files)
        )

        log = newLogger("model")
        value_outputs = self.handler.process(mode, values, files, OUTPUT_DIR, parameters, log)
        if value_outputs is None:
            value_outputs = {}

        output_json = self._build_output_json(value_outputs, output_path)
        self.runner_logger.info(
            "Built Output JSON", value_fields=len(value_outputs), assets=len(output_json) - len(value_outputs)
        )

        # Write value outputs to result.json so the orchestrator can read them
        result_path = OUTPUT_DIR / "result.json"
        result_path.write_text(json.dumps(value_outputs))

        print(json.dumps(output_json))

    def _parse_args(self, args: list[str] | None) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--parameters", required=True, help="JSON exec config from API"
        )
        parser.add_argument(
            "--local", action="store_true", help="Run locally without S3"
        )
        return parser.parse_args(args)

    def _resolve_inputs(self, inputs: dict) -> tuple[dict, dict[str, tuple[Path, str]]]:
        """Split raw inputs into value inputs and file inputs by wire shape.

        Wire shapes:
          {"field": {"value": X}}        -> value input   -> values["field"] = X
          {"field": {"uri": U, "mime_type": M}}
                                         -> file input    -> files["field"] = (path, M)
          {"field": "file://..."}        -> file input (local URI as a plain string)

        A field that is neither wrapped nor a URI is treated as a plain value.
        File inputs are downloaded from S3 (remote mode) or resolved from
        file:// URIs (local mode).
        """
        values: dict = {}
        local_files: dict[str, tuple[Path, str]] = {}
        remote_specs: dict[str, dict] = {}  # for StorageService.download_inputs

        for field_name, spec in inputs.items():
            if isinstance(spec, dict) and "value" in spec:
                values[field_name] = spec["value"]
            elif isinstance(spec, dict) and "uri" in spec:
                uri = spec["uri"]
                mime = spec.get("mime_type", "application/octet-stream")
                if self.local_mode or uri.startswith("file://"):
                    local_files[field_name] = (
                        self._resolve_local_file(uri),
                        mime,
                    )
                else:
                    remote_specs[field_name] = spec
            elif isinstance(spec, str) and (
                spec.startswith("file://") or spec.startswith("s3://")
            ):
                if self.local_mode or spec.startswith("file://"):
                    local_files[field_name] = (
                        self._resolve_local_file(spec),
                        "application/octet-stream",
                    )
                else:
                    remote_specs[field_name] = {
                        "uri": spec,
                        "mime_type": "application/octet-stream",
                    }
            else:
                # Plain value passed directly (not API-wrapped)
                values[field_name] = spec

        files: dict[str, tuple[Path, str]] = dict(local_files)
        if remote_specs and not self.local_mode and self.storage is not None:
            INPUT_DIR.mkdir(parents=True, exist_ok=True)
            downloaded = self.storage.download_inputs(remote_specs, INPUT_DIR)
            files.update(downloaded)

        return values, files

    def _resolve_local_file(self, uri: str) -> Path:
        """Convert file:// URI to local Path."""
        if uri.startswith("file://"):
            return Path(uri[7:])
        return Path(uri)

    def _build_output_json(self, value_outputs: dict, output_path: str) -> dict:
        """Merge value outputs with resolved asset references.

        Every file in OUTPUT_DIR is treated as an asset the handler produced.
        Asset field name == file name (the convention): the handler writes a
        file named exactly after the asset field declared in model.json. The
        runner uploads each file and emits ``{filename: {"url": ..., "mime_type": ...}}``.
        """
        output = dict(value_outputs)

        if not OUTPUT_DIR.exists():
            return output

        for file_path in sorted(OUTPUT_DIR.iterdir()):
            if not file_path.is_file():
                continue
            url = self._asset_url(file_path.name, output_path)
            mime, _ = mimetypes.guess_type(file_path.name)
            output[file_path.name] = {
                "url": url,
                "mime_type": mime or "application/octet-stream",
            }

        return output

    def _asset_url(self, filename: str, output_path: str) -> str:
        """Build the storage URL for an asset file, uploading it in remote mode."""
        if self.local_mode or self.storage is None or self.config is None:
            return f"file://{OUTPUT_DIR / filename}"

        bucket = self.config.bucket
        prefix = output_path.replace(f"s3://{bucket}/", "").strip("/")
        key = f"{prefix}/{filename}" if prefix else filename
        self.storage.s3.upload(OUTPUT_DIR / filename, key)
        self.runner_logger.debug("uploaded asset", file=filename, key=key)
        return f"s3://{bucket}/{key}"


def run(handler: ModelHandler, args: list[str] | None = None) -> None:
    """Convenience function to run a model handler."""
    # Parse args once here to decide local vs remote
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--local", action="store_true")
    parsed, _ = parser.parse_known_args(args)

    if parsed.local:
        runner = ModelRunner(handler, local_mode=True)
    else:
        config = S3Config.from_env()
        runner = ModelRunner(handler, local_mode=False, config=config)
    runner.run(args)
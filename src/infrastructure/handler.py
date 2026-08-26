from pathlib import Path
from typing import Protocol


class ModelHandler(Protocol):
    """Protocol for model handlers.

    Users implement a single ``process`` method. The runner selects the Mode by
    name, resolves value inputs and file inputs separately, and calls
    ``process``. The handler returns its *value* outputs as a dict and writes
    any *asset* (file) outputs to the output directory it receives, named
    exactly after the asset field declared in ``model.json``. The runner uploads
    those files and merges their URLs into the Output JSON.

    Example::

        class MyModel:
            def process(self, mode, values, files, output_dir, parameters, logger):
                if mode == "instance":
                    # value inputs come unwrapped: values["feature1"] == 3.0
                    return {"prediction": values["feature1"] * parameters["multiplier"]}
                elif mode == "dataset":
                    # file inputs come as {field: (local_path, mime_type)}
                    path, _ = files["file"]
                    ...
                    # asset output: write a file named after the declared asset field
                    (output_dir / "result.json").write_text(...)
                    return {}  # no value outputs; the asset is the output
    """

    def process(
        self,
        mode: str,
        values: dict,
        files: dict[str, tuple[Path, str]],
        output_dir: Path,
        parameters: dict,
        logger,
    ) -> dict:
        """Run the model for the selected Mode.

        Args:
            mode: The Mode name selected by the caller (e.g. "instance",
                "dataset", or any custom name declared in ``model.json``).
            values: Value inputs, already unwrapped from the API wire format
                (``{"field": {"value": X}}`` -> ``{"field": X}``). Empty when the
                Mode has no value inputs.
            files: File inputs, already resolved to local paths as
                ``{field_name: (local_path, mime_type)}``. Empty when the Mode
                has no file inputs. The handler never touches storage.
            output_dir: Directory to write asset (file) outputs to. The handler
                writes each asset file named exactly after the asset field
                declared in ``model.json``; the runner uploads them and merges
                their URLs into the Output JSON.
            parameters: Model-level parameters shared across all Modes.
            logger: structlog logger with run context bound.

        Returns:
            A dict of value outputs. These become the value fields of the Output
            JSON. To produce asset (file) outputs, write files to ``output_dir``
            named exactly after the asset field declared in ``model.json``; the
            runner uploads them and merges their URLs into the Output JSON.
        """
        ...
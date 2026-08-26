import argparse
import json
import sys
from pathlib import Path

import jsonschema


def _check(fails: list, ok: bool, label: str) -> None:
    print(f"{'OK  ' if ok else 'FAIL'} {label}")
    if not ok:
        fails.append(label)


def _expect_invalid(fails: list, validator, label: str, obj: dict) -> None:
    errs = list(validator.iter_errors(obj))
    _check(fails, bool(errs), f"reject: {label}")
    if errs:
        print(f"     rejected: {errs[0].message}")


def _expect_valid(fails: list, validator, label: str, obj: dict) -> None:
    errs = list(validator.iter_errors(obj))
    _check(fails, not errs, f"accept: {label}")
    if errs:
        print(f"     error: {errs[0].message}")


def _run_probes(validator: jsonschema.Draft202012Validator) -> list:
    """Run edge-case probes beyond basic schema validation."""
    fails: list = []

    # --- file type rules ---
    _expect_invalid(fails, validator, "file output missing mime_type",
        {"name":"m","parameters":{},"execution":{"modes":{"x":{"outputs":{"a":{"type":"file"}}}}}})
    _expect_valid(fails, validator, "file output with mime_type",
        {"name":"m","parameters":{},"execution":{"modes":{"x":{"outputs":{"a":{"type":"file","mime_type":"text/csv"}}}}}})
    _expect_valid(fails, validator, "image/* wildcard",
        {"name":"m","parameters":{},"execution":{"modes":{"x":{"outputs":{"a":{"type":"file","mime_type":"image/*"}}}}}})
    _expect_invalid(fails, validator, "*/* wildcard",
        {"name":"m","parameters":{},"execution":{"modes":{"x":{"outputs":{"a":{"type":"file","mime_type":"*/*"}}}}}})
    _expect_invalid(fails, validator, "partial wildcard image/pn*",
        {"name":"m","parameters":{},"execution":{"modes":{"x":{"outputs":{"a":{"type":"file","mime_type":"image/pn*"}}}}}})
    _expect_invalid(fails, validator, "file field carrying items",
        {"name":"m","parameters":{},"execution":{"modes":{"x":{"outputs":{"a":{"type":"file","mime_type":"text/csv","items":{"type":"string"}}}}}}})

    # --- mode structural rules ---
    _expect_invalid(fails, validator, "mode missing outputs",
        {"name":"m","parameters":{},"execution":{"modes":{"x":{"inputs":{"a":{"type":"integer"}}}}}})
    _expect_valid(fails, validator, "mode no inputs, has outputs (parameter-only)",
        {"name":"m","parameters":{},"execution":{"modes":{"x":{"outputs":{"a":{"type":"integer"}}}}}})
    _expect_invalid(fails, validator, "execution with no modes",
        {"name":"m","parameters":{},"execution":{"modes":{}}})

    # --- array/object rules ---
    _expect_invalid(fails, validator, "array field without items or prefixItems",
        {"name":"m","parameters":{},"execution":{"modes":{"x":{"inputs":{"a":{"type":"array"}},"outputs":{"o":{"type":"integer"}}}}}})
    _expect_valid(fails, validator, "array field with prefixItems only (tuple)",
        {"name":"m","parameters":{},"execution":{"modes":{"x":{"inputs":{"a":{"type":"array","prefixItems":[{"type":"number"},{"type":"number"}]}},"outputs":{"o":{"type":"integer"}}}}}})

    # --- response rule ---
    _expect_invalid(fails, validator, "response bogus value",
        {"name":"m","parameters":{},"execution":{"modes":{"x":{"response":"async","outputs":{"a":{"type":"integer"}}}}}})

    # --- instance/dataset not reserved ---
    _expect_valid(fails, validator, "custom mode name 'render'",
        {"name":"m","parameters":{},"execution":{"modes":{"render":{"response":"deferred","outputs":{"video":{"type":"file","mime_type":"video/mp4"}}}}}})

    return fails


def main():
    parser = argparse.ArgumentParser(description="Validate model.json against schema")
    parser.add_argument(
        "model_path",
        nargs="?",
        default="schema/model.json",
        help="Path to model.json (default: schema/model.json)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    schema_path = model_path.parent / "model.schema.json"

    if not model_path.exists():
        print(f"Error: {model_path} not found", file=sys.stderr)
        sys.exit(1)

    if not schema_path.exists():
        print(f"Error: {schema_path} not found", file=sys.stderr)
        sys.exit(1)

    schema = json.loads(schema_path.read_text())
    model = json.loads(model_path.read_text())

    validator = jsonschema.Draft202012Validator(schema)

    # 1. Schema validation
    errors = list(validator.iter_errors(model))
    if errors:
        for error in errors:
            path = ".".join(str(p) for p in error.path) or "root"
            print(f"Validation error at {path}: {error.message}")
        sys.exit(1)

    # 2. Required top-level fields
    for field in ("name", "execution"):
        if field not in model:
            print(f"Error: model.json missing required field '{field}'", file=sys.stderr)
            sys.exit(1)

    print("Schema validation passed!")

    # 3. Edge-case probes
    print("\nRunning edge-case probes...")
    probe_fails = _run_probes(validator)

    print(f"\n{'ALL PASSED' if not probe_fails else f'{len(probe_fails)} PROBE FAILURE(S)'}")
    sys.exit(1 if probe_fails else 0)


if __name__ == "__main__":
    main()
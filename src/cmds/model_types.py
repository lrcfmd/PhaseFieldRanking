#!/usr/bin/env python3
"""Generate Python types from model.json schema."""

import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict, Literal


TYPE_MAP = {
    "string": "str",
    "number": "float",
    "integer": "int",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
    # file inputs arrive as (local_path, mime_type) tuples, already resolved
    # by the runner. file outputs are assets the handler writes to disk, not
    # values it returns, so they don't appear in generated input TypedDicts.
    "file": "tuple[Path, str]",
}

INDENT = "    "


def to_class_name(key: str) -> str:
    """Convert a key to a PascalCase class name."""
    return "".join(word.capitalize() for word in key.replace("-", "_").split("_"))


def get_python_type(prop: dict) -> str:
    """Get Python type for a property."""
    if "$ref" in prop:
        return prop["$ref"].split("/")[-1]

    type_str = prop.get("type", "Any")

    if isinstance(type_str, list):
        types = [TYPE_MAP.get(t, t) for t in type_str]
        if len(types) == 1:
            type_str = types[0]
        else:
            return f"Union[{', '.join(types)}]"

    python_type = TYPE_MAP.get(type_str, type_str)

    if "enum" in prop and isinstance(prop["enum"], list):
        if len(prop["enum"]) == 0:
            return "Any "
        # Get the Python representation of each enum value
        enum_values = []
        for v in prop["enum"]:
            if isinstance(v, str):
                enum_values.append(repr(v))
            elif isinstance(v, bool):
                enum_values.append(str(v))
            else:
                enum_values.append(repr(v))
        return f"Literal[{', '.join(enum_values)}]"

    if type_str == "array":
        if "items" in prop:
            items_type = get_python_type(prop["items"])
            return f"list[{items_type}]"
        return "list"

    if type_str == "object":
        if "properties" in prop:
            return "dict"

    return python_type


def generate_typed_dict(
    class_name: str, properties: dict, nested_defs: dict
) -> list[str]:
    """Generate a TypedDict class."""
    lines = [f"class {class_name}(TypedDict, total=False):"]

    for key, prop in properties.items():
        if key in (
            "description",
            "type",
            "additionalProperties",
            "minProperties",
            "maxProperties",
        ):
            continue

        python_type = get_python_type(prop)
        description = prop.get("description", "")
        default = prop.get("default")

        # Check if this is a nested object with its own properties
        if prop.get("type") == "object" and "properties" in prop:
            nested_name = to_class_name(key)
            nested_defs[nested_name] = prop["properties"]

        if description:
            lines.append(f"{INDENT}# {description}")
        if default is not None:
            lines.append(f"{INDENT}{key}: {python_type}  # default: {default!r}")
        else:
            lines.append(f"{INDENT}{key}: {python_type}")

    return lines


def generate_types(model_path: Path) -> str:
    """Generate Python types from model.json."""
    model = json.loads(model_path.read_text())

    header = [
        "# Auto-generated from model.json - DO NOT EDIT",
        "# Regenerate with: uv run model-types <path-to-model.json> -o src/model/types.py",
        "",
        "from pathlib import Path",
        "from typing import TypedDict, Literal",
        "",
    ]

    all_classes: list[str] = []
    nested_defs: dict = {}

    # Generate Parameters
    parameters = model.get("parameters", {})
    if parameters:
        all_classes.extend(generate_typed_dict("Parameters", parameters, nested_defs))

    # Generate Execution classes — one input TypedDict per Mode that declares
    # inputs. The class name is <ModeName>Inputs (e.g. InstanceInputs,
    # DatasetInputs), derived from the Mode key.
    execution = model.get("execution", {})
    modes = execution.get("modes", {})

    for mode_name, mode_def in modes.items():
        mode_inputs = mode_def.get("inputs", {})
        if mode_inputs:
            if all_classes:
                all_classes.append("")
            class_name = f"{to_class_name(mode_name)}Inputs"
            all_classes.extend(
                generate_typed_dict(class_name, mode_inputs, nested_defs)
            )

    # Generate nested classes
    nested_classes: list[str] = []
    for name, props in nested_defs.items():
        if nested_classes:
            nested_classes.append("")
        nested_classes.extend(generate_typed_dict(name, props, {}))

    # Combine
    if nested_classes:
        all_classes.append("")
        all_classes.extend(nested_classes)

    return "\n".join(header + all_classes)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Python types from model.json"
    )
    parser.add_argument("model_path", help="Path to model.json")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: {model_path} not found", file=sys.stderr)
        sys.exit(1)

    types_output = generate_types(model_path)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(types_output)
        print(f"Generated types written to {output_path}")
    else:
        print(types_output)


if __name__ == "__main__":
    main()

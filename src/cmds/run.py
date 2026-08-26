from infrastructure.runner import run
from model import handler


def run_cli() -> None:
    """CLI entry point for model-run command."""
    run(handler)


if __name__ == "__main__":
    run_cli()
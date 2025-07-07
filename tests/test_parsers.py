import sys
import yaml
import tempfile
from pathlib import Path
import pandas as pd
from dataclasses import asdict
from ranking_phase_fields.parse_icsd import parse_input, parse_icsd
from ranking_phase_fields.logger import get_logger

sys.path.append(str(Path(__file__).resolve().parent.parent))
logger = get_logger(__name__)

def test_parse_input_rpp_yaml():
    """Test parse_input with example rpp.yaml file."""

    example_yaml = {
        "phase_fields": "quaternary",
        "anions_train": ["S", "O", "Cl", "Br"],
        "nanions_train": 2,
        "cations_train": "all",
        "icsd_file": "icsd2017",
    }

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as f:
        yaml.dump(example_yaml, f)
        yaml_path = f.name

    params = parse_input(yaml_path)

    assert params["phase_fields"] == "quaternary"
    assert isinstance(params["anions_train"], list)
    assert params["nanions_train"] == 2
    assert params["icsd_file"] == "icsd2017"

    Path(yaml_path).unlink()

def test_parse_icsd_file(tmp_path):
    """Test parse_icsd of icsd2017 file """

    # Example usage
    fields = parse_icsd(
        phase_fields="ternary",
        anions_train=["O", "F"],
        nanions_train=1,
        cations_train=["Cr", "Te", "Mn", "Si", "La", "Co", "Li", "Si", "Fe"],
        icsd_file='icsd2017'
    )

    assert isinstance(fields, list)
    assert all(isinstance(f, list) for f in fields)
    assert any("O" in f or "F" in f for f in fields)
    assert len(fields) > 0

"""
def test_parse_icsd_custom_csv(tmp_path):
#    "Test parse_icsd when using a CSV file instead of ICSD file."

    csv_file = tmp_path / "custom_phasefields.csv"
    df = pd.DataFrame({
        "field": ["Na Cl", "Li O S", "K Br I"]
    })
    df.to_csv(csv_file, index=False)

    fields = parse_icsd(
        phase_fields="binary",
        anions_train=["Cl", "O", "S", "Br", "I"],
        nanions_train=0,
        cations_train=["Na", "Li", "K"],
        icsd_file=str(csv_file.name)
    )

    assert isinstance(fields, list)
    assert all(isinstance(f, list) for f in fields)
    assert len(fields) == 3
"""

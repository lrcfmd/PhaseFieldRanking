import csv
from pathlib import Path

import pytest

from infrastructure.logging import newLogger
from model.model import PhaseFieldRankingModel

LOG = newLogger("test")


@pytest.fixture(scope="module")
def model() -> PhaseFieldRankingModel:
    return PhaseFieldRankingModel()


def test_unknown_mode(model: PhaseFieldRankingModel) -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        model.process("bogus", {}, {}, Path("/tmp/pfr_out"), {}, LOG)


def test_parse_rankings(tmp_path: Path) -> None:
    csv_path = tmp_path / "r.csv"
    csv_path.write_text(
        "Phase fields,scores,Norm. scores\n"
        "Li Mg O        , 1.323,  1.323\n"
        "Li Ca O        , 0.501,  0.379\n"
    )
    records = PhaseFieldRankingModel._parse_rankings(csv_path)
    assert len(records) == 2
    assert records[0]["phase_field"] == "Li Mg O"
    assert records[0]["score"] == pytest.approx(1.323)


def test_rank_ternary(model: PhaseFieldRankingModel, tmp_path: Path) -> None:
    """End-to-end: trains a small VAE and ranks a tiny ternary study."""
    out = tmp_path / "output"
    result = model.process(
        "rank", {}, {}, out,
        {
            "phase_fields": "ternary",
            "method": "VAE",
            "features": "magpie",
            "nanions_train": 1,
            "anions_train": ["O", "S"],
            "cation1_test": ["Li"],
            "cation2_test": ["Mg"],
            "anion1_test": ["O"],
        },
        LOG,
    )
    # rankings.csv asset written
    rankings = out / "rankings.csv"
    assert rankings.exists()
    with rankings.open() as f:
        header = next(csv.reader(f))
    assert header[0].strip().lower().startswith("phase fields")

    # value outputs
    assert result["count"] >= 1
    assert len(result["top_ranked"]) >= 1
    top = result["top_ranked"][0]
    assert set(top) == {"phase_field", "score", "norm_score"}
    # the requested Li-Mg-O phase field should appear
    assert any("Li" in r["phase_field"] for r in result["top_ranked"])


# ─── Input checks (all fail before any model is trained) ────────────────────

from infrastructure.handler import InputError
from model.model import DEFAULTS, validate_config

TERNARY = {
    "phase_fields": "ternary",
    "nanions_train": 1,
    "anions_train": ["O", "S"],
    "cation1_test": ["Li"],
    "cation2_test": ["Mg"],
    "anion1_test": ["O"],
}


def _cfg(**over) -> dict:
    return {**DEFAULTS, **TERNARY, **over}


@pytest.mark.parametrize(
    "over, match",
    [
        ({"cation1_test": ["li"]}, "did you mean 'Li'"),
        ({"anion1_test": ["O", "Xx"]}, "'Xx' is not an element symbol"),
        ({"anions_train": ["Oxygen"]}, "'Oxygen' is not an element symbol"),
        ({"cations_train": "Li, Qq"}, "'Qq' is not an element symbol"),
        ({"phase_fields": "quinary"}, "quinary"),
        ({"features": "nonexistent"}, "Unknown feature set"),
        ({"icsd_file": "my_data.csv"}, "Unknown training data file"),
        ({"nanions_train": 3}, "nanions_train must be 0, 1 or 2"),
        ({"average_runs": 0}, "average_runs must be a whole number of at least 1"),
        ({"average_runs": 1.5}, "average_runs must be a whole number"),
        ({"anion1_test": []}, "anion1_test is empty"),
        ({"cation2_test": []}, "cation2_test is empty"),
        ({"nanions_train": 2, "anion2_test": []}, "anion2_test is empty"),
        ({"nanions_train": 1, "anions_train": []}, "anions_train is empty"),
        ({"cation1_test": ["Es"]}, "'magpie' feature set has no data for Es"),
    ],
)
def test_validate_config_rejects(over: dict, match: str) -> None:
    with pytest.raises(InputError, match=match):
        validate_config(_cfg(**over))


def test_validate_config_normalises_lists() -> None:
    cfg = validate_config(_cfg(cations_train="Li, Na Mg", anion1_test=["O, S", "O"]))
    assert cfg["cations_train"] == ["Li", "Na", "Mg"]
    assert cfg["anion1_test"] == ["O", "S"]
    # Positions not used by the chosen layout may be left empty.
    cfg = validate_config(_cfg(nanions_train=2, cation2_test=[], anion2_test=["S"]))
    assert cfg["cation2_test"] == []
    assert validate_config(_cfg(cations_train="ALL"))["cations_train"] == "all"


def test_process_reports_input_error(model: PhaseFieldRankingModel, tmp_path: Path) -> None:
    with pytest.raises(InputError, match="did you mean 'Mg'"):
        model.process("rank", {}, {}, tmp_path, {**TERNARY, "cation2_test": ["MG"]}, LOG)


def test_no_phase_fields_to_rank(model: PhaseFieldRankingModel, tmp_path: Path) -> None:
    # Every combination repeats Mg, so there is nothing to rank.
    params = {**TERNARY, "cation2_test": ["Mg"], "anion1_test": ["Mg"]}
    with pytest.raises(InputError, match="no new phase fields to rank"):
        model.process("rank", {}, {}, tmp_path, params, LOG)


def test_empty_training_set(model: PhaseFieldRankingModel, tmp_path: Path) -> None:
    params = {**TERNARY, "cations_train": "He", "anions_train": ["O"]}
    with pytest.raises(InputError, match="No ternary phase fields in the training data"):
        model.process("rank", {}, {}, tmp_path, params, LOG)


def test_schema_enums_match_bundled_data() -> None:
    import json

    from model.model import available_features

    schema = json.loads((Path(__file__).parents[1] / "schema" / "model.json").read_text())
    params = schema["parameters"]
    assert params["features"]["enum"] == available_features()
    for name in params["icsd_file"]["enum"]:
        validate_config(_cfg(icsd_file=name))

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

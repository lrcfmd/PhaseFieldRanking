import csv
import glob
import os
import shutil
import tempfile
from pathlib import Path

import yaml

from infrastructure.logging import newLogger
from ranking_phase_fields.__main__ import PhaseFieldRanker

# Defaults mirror the repository's config.yaml. Any parameter supplied by the
# caller overrides the corresponding default.
DEFAULTS = {
    "icsd_file": "icsd2017",
    "phase_fields": "quaternary",
    "cations_train": "all",
    "anions_train": ["S", "O", "Cl", "Br", "F", "N", "Te", "P", "Se", "As", "I"],
    "nanions_train": 2,
    "cation1_test": ["Li"],
    "cation2_test": ["B", "Mg", "Al", "Si", "P", "K", "Ca", "Zn", "Sr", "Y",
                     "Zr", "Sn", "Ba", "Ta", "La"],
    "anion1_test": ["S", "O", "Cl", "Br", "I", "N", "F"],
    "anion2_test": ["S", "O", "Cl", "Br", "I", "N", "F"],
    "method": "VAE",
    "cross_validate": False,
    "average_runs": 1,
    "features": "magpie",
}


class PhaseFieldRankingModel:
    """Rank unexplored phase fields by likelihood of hosting a stable compound.

    Wraps the ``ranking_phase_fields`` package (PhaseRank): an unsupervised
    anomaly-detection model (PyOD AE/VAE, or a Keras VAE encoder) is trained at
    run time on ICSD phase-field patterns, and the requested unexplored phase
    fields are scored by similarity to known chemistry. Reference data (ICSD
    excerpt, elemental features, tables) is bundled inside the package.

    Single Mode ``rank`` (deferred): the study is defined entirely by
    model-level parameters; the ranked phase fields are written to a CSV asset.
    """

    def __init__(self):
        self.logger = newLogger("model")

    def process(
        self,
        mode: str,
        values: dict,
        files: dict[str, tuple[Path, str]],
        output_dir: Path,
        parameters: dict,
        logger,
    ) -> dict:
        if mode != "rank":
            raise ValueError(f"Unknown mode: {mode!r}")

        # Merge caller parameters over the defaults (ignore unset/None).
        cfg = dict(DEFAULTS)
        for k, v in (parameters or {}).items():
            if v is not None:
                cfg[k] = v

        logger.info(
            "ranking phase fields",
            phase_fields=cfg["phase_fields"],
            method=cfg["method"],
            features=cfg["features"],
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        # The pipeline writes its result CSVs to the current working directory,
        # so run it inside a temp dir and collect the test-scores file.
        prev_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            config_path = tmp_dir / "config.yaml"
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

            os.chdir(tmp_dir)
            try:
                ranker = PhaseFieldRanker(str(config_path))
                ranker.run_study()
            finally:
                os.chdir(prev_cwd)

            matches = sorted(glob.glob(str(tmp_dir / "*_test_scores.csv")))
            if not matches:
                raise RuntimeError("ranking produced no test-scores output")
            rankings_path = output_dir / "rankings.csv"
            shutil.copyfile(matches[0], rankings_path)

        ranked = self._parse_rankings(rankings_path)
        logger.info("wrote asset", path=str(rankings_path), ranked=len(ranked))

        # Lower score = smaller reconstruction error = more similar to known
        # chemistry = more promising. Surface the most promising inline.
        ranked_sorted = sorted(ranked, key=lambda r: r["score"])
        return {
            "count": len(ranked_sorted),
            "top_ranked": ranked_sorted[:20],
        }

    @staticmethod
    def _parse_rankings(path: Path) -> list[dict]:
        """Parse the `Phase fields, scores, Norm. scores` CSV into records."""
        records: list[dict] = []
        with path.open(newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    records.append({
                        "phase_field": row[0].strip(),
                        "score": float(row[1]),
                        "norm_score": float(row[2]),
                    })
                except ValueError:
                    continue
        return records

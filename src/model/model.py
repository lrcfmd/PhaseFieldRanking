import csv
import glob
import os
import shutil
import tempfile
from pathlib import Path

import yaml

from infrastructure.handler import InputError
from infrastructure.logging import newLogger
import ranking_phase_fields
from ranking_phase_fields.__main__ import PhaseFieldRanker
from ranking_phase_fields.parse_icsd import ELEMENT_SYMBOLS

PACKAGE_DATA = Path(ranking_phase_fields.__file__).parent / "data"
FEATURES_DIR = PACKAGE_DATA / "elemental_features"

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
        cfg = validate_config(cfg)

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
                # Build the training and testing phase fields first (seconds),
                # so impossible studies fail before the model is trained.
                ranker.prepare()
                check_study(cfg, ranker.training, ranker.testing)
                ranker.train_and_rank()
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


# ─── Input checks ───────────────────────────────────────────────────────────

# Which element lists define each position of the phase fields to rank. This
# mirrors ranking_phase_fields.parse_icsd.parse_input.
def ranking_positions(phase_fields: str, nanions_train: int) -> list[str]:
    if phase_fields == "ternary" and nanions_train == 1:
        return ["cation1_test", "cation2_test", "anion1_test"]
    if phase_fields == "ternary":
        return ["cation1_test", "anion1_test", "anion2_test"]
    return ["cation1_test", "cation2_test", "anion1_test", "anion2_test"]


def available_features() -> list[str]:
    return sorted(p.stem for p in FEATURES_DIR.glob("*.csv"))


def feature_elements(features: str) -> set[str]:
    with (FEATURES_DIR / f"{features}.csv").open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return {row[0].strip() for row in reader if row}


def _whole_number(value):
    """An integer given as int, integral float or numeric string; else None."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else None


def _element_list(name: str, value) -> list[str]:
    """Read an element list (a list, or a string separated by commas/spaces) and
    check every entry is an element symbol."""
    if value is None:
        return []
    if isinstance(value, str):
        items = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple)):
        items = []
        for v in value:
            items += str(v).replace(",", " ").split()
    else:
        raise InputError(f"{name} must be a list of element symbols, e.g. Li, Mg, O.")
    symbols = []
    for item in items:
        if item not in ELEMENT_SYMBOLS:
            hint = item[:1].upper() + item[1:].lower()
            if hint in ELEMENT_SYMBOLS:
                raise InputError(
                    f"{name}: '{item}' is not an element symbol - did you mean '{hint}'? "
                    "Symbols are case-sensitive."
                )
            raise InputError(
                f"{name}: '{item}' is not an element symbol. Use symbols such as Li, Mg, O."
            )
        if item not in symbols:
            symbols.append(item)
    return symbols


def validate_config(cfg: dict) -> dict:
    """Check the study parameters and normalise the element lists.

    Raises InputError, before any work is done, for anything the user can fix.
    """
    cfg = dict(cfg)

    phase_fields = cfg["phase_fields"]
    if phase_fields == "quinary":
        raise InputError(
            "Ranking quinary phase fields is not supported: the study can only be built "
            "for ternary or quaternary phase fields. Choose ternary or quaternary."
        )
    if phase_fields not in ("ternary", "quaternary"):
        raise InputError(
            f"phase_fields must be ternary or quaternary, got '{phase_fields}'."
        )

    if cfg["method"] not in ("VAE", "AE", "VAE_encoder"):
        raise InputError(f"method must be VAE, AE or VAE_encoder, got '{cfg['method']}'.")

    features = str(cfg["features"]).strip()
    if features not in available_features():
        raise InputError(
            f"Unknown feature set '{features}'. Choose one of: "
            f"{', '.join(available_features())}."
        )
    cfg["features"] = features

    icsd_file = str(cfg["icsd_file"]).strip()
    icsd_files = sorted(
        p.name for p in PACKAGE_DATA.iterdir() if p.is_file() and "icsd" in p.name.lower()
    )
    if icsd_file not in icsd_files:
        raise InputError(
            f"Unknown training data file '{icsd_file}'. Choose one of: {', '.join(icsd_files)}."
        )
    cfg["icsd_file"] = icsd_file

    nanions = _whole_number(cfg["nanions_train"])
    if nanions not in (0, 1, 2):
        raise InputError(f"nanions_train must be 0, 1 or 2, got '{cfg['nanions_train']}'.")
    cfg["nanions_train"] = nanions

    average_runs = _whole_number(cfg["average_runs"])
    if average_runs is None or average_runs < 1:
        raise InputError(
            f"average_runs must be a whole number of at least 1, got '{cfg['average_runs']}'."
        )
    cfg["average_runs"] = average_runs

    # Training-set definition.
    cations_train = cfg["cations_train"]
    if isinstance(cations_train, str) and cations_train.strip().lower() in ("", "all"):
        cfg["cations_train"] = "all"
    else:
        cfg["cations_train"] = _element_list("cations_train", cations_train)
        if not cfg["cations_train"]:
            raise InputError("cations_train is empty: give 'all' or a list of elements, e.g. Li, Na, Mg.")
    cfg["anions_train"] = _element_list("anions_train", cfg["anions_train"])
    if nanions > 0 and not cfg["anions_train"]:
        raise InputError(
            f"anions_train is empty, but nanions_train is {nanions}: the training phase "
            "fields need at least that many anions. List the anions, e.g. O, S, F, "
            "or set nanions_train to 0."
        )

    # Phase fields to rank.
    for name in ("cation1_test", "cation2_test", "anion1_test", "anion2_test"):
        cfg[name] = _element_list(name, cfg.get(name))
    positions = ranking_positions(phase_fields, nanions)
    for name in positions:
        if not cfg[name]:
            raise InputError(
                f"{name} is empty, but it is needed for {phase_fields} phase fields "
                f"with nanions_train = {nanions} (positions used: {', '.join(positions)}). "
                "Give at least one element."
            )
    known = feature_elements(features)
    missing = sorted({el for name in positions for el in cfg[name] if el not in known})
    if missing:
        raise InputError(
            f"The '{features}' feature set has no data for {', '.join(missing)}, so these "
            "elements cannot be ranked. Remove them or choose another feature set."
        )
    return cfg


def check_study(cfg: dict, training: list, testing: list) -> None:
    """Fail clearly when the study has nothing to train on or nothing to rank."""
    if not training:
        raise InputError(
            f"No {cfg['phase_fields']} phase fields in the training data match the training "
            f"settings (cations_train, anions_train = {', '.join(cfg['anions_train']) or 'none'}, "
            f"nanions_train = {cfg['nanions_train']}). Allow more elements or lower nanions_train."
        )
    if cfg["cross_validate"] and len(training) < 5:
        raise InputError(
            f"Cross-validation needs at least 5 training phase fields, but only "
            f"{len(training)} match the training settings. Turn cross_validate off "
            "or widen the training set."
        )
    if not testing:
        positions = ranking_positions(cfg["phase_fields"], cfg["nanions_train"])
        lists = "; ".join(f"{n} = {', '.join(cfg[n])}" for n in positions)
        raise InputError(
            f"There are no new phase fields to rank from {lists}. Each phase field needs "
            f"{len(positions)} different elements, one from each list, and must not already "
            "be in the training data. Add more elements to the lists."
        )

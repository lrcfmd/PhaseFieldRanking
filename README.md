# Phase Field Ranking (PhaseRank)

Ranks unexplored **phase fields** (combinations of chemical elements) by their
likelihood of hosting a stable compound, for the LMDS platform.

An unsupervised anomaly-detection model (PyOD AutoEncoder / Variational
AutoEncoder, or a Keras VAE encoder) is trained **at run time** on phase-field
patterns extracted from the ICSD, and the requested unexplored phase fields are
scored by similarity to that known chemistry — a lower reconstruction error is
hypothesised to indicate a more promising combination.

Based on: A. Vasylenko et al., *"Element selection for crystalline inorganic
solid discovery guided by unsupervised machine learning of experimentally
explored chemistry"*, Nat. Commun. 12 (2021). This is a conversion of the
`ranking_phase_fields` package to the LMDS model-template contract; the package
and its reference data are vendored under `src/`.

## Quick Start

```bash
# Small, quick ternary study (trains a VAE, ~tens of seconds):
scripts/run_rank.sh          # -> data/output/rankings.csv
```

## Mode

Single mode **`rank`** (deferred — it trains a model, so it is not instant). The
study is defined entirely by **parameters**; there are no per-run file inputs.

| Output | Meaning |
|--------|---------|
| `rankings.csv` (asset) | Ranked phase fields with raw and normalised scores |
| `count` (value) | Number of phase fields ranked |
| `top_ranked` (value) | The most promising phase fields (lowest score first) |

### Key parameters (full list + defaults in `schema/model.json`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `phase_fields` | `quaternary` | `ternary` / `quaternary` / `quinary` |
| `method` | `VAE` | `VAE`, `AE` (PyOD/torch) or `VAE_encoder` (Keras) |
| `features` | `magpie` | Elemental feature set (see `data/elemental_features/`) |
| `cation1_test` … `anion2_test` | Li / … | Element lists defining the phase fields to rank |
| `anions_train`, `nanions_train`, `cations_train` | — | Definition of the ICSD training set |
| `cross_validate`, `average_runs` | `false`, `1` | Validation / averaging controls |

Lower score ⇒ smaller reconstruction error ⇒ more similar to known chemistry ⇒
more promising.

## Model data (bundled)

Unlike the other LMDS tools, PhaseRank has **no pre-trained model weights** — it
trains at run time. Its ~11 MB of reference data (the ICSD excerpt, elemental
feature sets and Magpie tables) is **bundled inside the package** at
`src/ranking_phase_fields/data/` and ships in the wheel/image; there are no
runtime assets to upload.

## Dependencies

Runs on **Python 3.10** (the bundled package targets `<3.11`). The stack is
CPU-only: `pyod` + `torch` (AE/VAE), `tensorflow-cpu` (the `VAE_encoder`
method's Keras model), plus `numpy`/`pandas`/`pydantic`/`pyyaml`. It is a heavy
image (torch + TensorFlow).

## Local development

```bash
uv sync --extra dev
uv run validate-model schema/model.json
uv run pytest tests/ -v          # includes one slow end-to-end training test
```

## Docker

```bash
docker build -t phase-field-ranking:1.0.0 .
docker run --rm -e DATA_DIR=/data phase-field-ranking:1.0.0 --local \
  --parameters '{"mode":"rank","inputs":{},"parameters":{"phase_fields":"ternary","method":"VAE","features":"magpie","nanions_train":1,"anions_train":["O","S"],"cation1_test":["Li"],"cation2_test":["Mg","Ca"],"anion1_test":["O"]}}'
```

## Environment Variables

Copy `.env.example` to `.env` and adjust for remote (S3) runs:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `DEBUG` | Logging verbosity |
| `S3_ENDPOINT` | `http://minio:9000` | S3-compatible storage endpoint |
| `S3_ACCESS_KEY` | `minioadmin` | S3 access key |
| `S3_SECRET_KEY` | `minioadmin` | S3 secret key |
| `S3_BUCKET` | `lmds` | S3 bucket name |

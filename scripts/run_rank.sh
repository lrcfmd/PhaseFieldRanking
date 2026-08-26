#!/usr/bin/env bash
set -euo pipefail

# Rank unexplored phase fields (no S3 required). The study is defined entirely
# by parameters. This example runs a small, quick ternary study; edit the
# parameters below (or pass your own) for larger runs.
#
# Note: the model trains at run time, so this takes tens of seconds+ depending
# on phase_fields and the size of the test element lists.

uv run model-run --local \
  --parameters "$(jq -n '{
    mode: "rank",
    inputs: {},
    parameters: {
      phase_fields: "ternary",
      method: "VAE",
      features: "magpie",
      nanions_train: 1,
      anions_train: ["O","S"],
      cation1_test: ["Li"],
      cation2_test: ["Mg","Ca"],
      anion1_test: ["O"]
    }
  }')"
echo "Rankings written to ./data/output/rankings.csv"

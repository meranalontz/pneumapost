#!/usr/bin/env bash
set -euo pipefail

# Eight-way data-parallel dispatch run on a single node (paper compute profile).
RUN=${1:-circulars/runs/main.conf}
OUT=${2:-runs/pneumapost.pt}

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  -m pneumapost.wicket train "${RUN}" --out "${OUT}" device=cuda

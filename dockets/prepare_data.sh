#!/usr/bin/env bash
set -euo pipefail

# Stage auxiliary public layers under data/raw/. Adapters expect tab-separated
# matrices; see datasets.txt for the canonical sources and access terms.
DEST=${1:-data/raw}
mkdir -p "${DEST}"

cat <<'NOTE'
Place the following under data/raw/ before invoking the omics / 16S adapters:
  tcga_luad_omics.tsv   samples x features, header row, first column = sample id
  oral16s_genus.tsv     samples x genera, header row, first column = sample id
The private Layer-1 cohort is not redistributable; the synthetic generator is
used by default and needs no downloaded files.
NOTE

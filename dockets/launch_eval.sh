#!/usr/bin/env bash
set -euo pipefail

RUN=${1:-circulars/runs/main.conf}
ARM=${2:-prospective}

python -m pneumapost.wicket appraise "${RUN}" --arm "${ARM}"

#!/bin/bash
# 2026-10-14: the paper's GPU work in sequence: O6 (pilot-gated, scripts/run_o6_queue.sh), then R6
# (scripts/run_r6.sh). Each stage takes the shared GPU lock itself, so other sessions interleave.
cd "$(dirname "$0")/.."
scripts/run_o6_queue.sh
scripts/run_r6.sh

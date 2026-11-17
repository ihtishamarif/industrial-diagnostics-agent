#!/usr/bin/env bash
# Record the real-board demo (Uno + LED + LDR) as a terminal GIF and keep the before/after plot.
# Needs: asciinema, agg (https://github.com/asciinema/agg), the board on $PORT, constant room light.
# Usage: scripts/record_real_demo.sh [/dev/ttyACM0]
set -euo pipefail
PORT="${1:-/dev/ttyACM0}"
cd "$(dirname "$0")/.."
unset PYTHONPATH AMENT_PREFIX_PATH
for tool in asciinema agg; do
  command -v "$tool" >/dev/null || { echo "missing: $tool"; exit 1; }
done
source .venv/bin/activate
asciinema rec --overwrite --cols 110 --rows 32 docs/real_board.cast \
  -c "diagagent demo --device serial --serial-port $PORT --backend offline --approve yes"
agg --speed 2 docs/real_board.cast docs/real_board.gif
cp results/pid_before_after.png docs/real_board_before_after.png
ls -l docs/real_board.gif docs/real_board_before_after.png

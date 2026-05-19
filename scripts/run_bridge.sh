#!/usr/bin/env bash
# Auto-restart wrapper for the Tender Agent bridge.
#
# Why this exists: macOS sometimes SIGKILLs the bridge under memory pressure,
# or a stray `pkill` from another terminal will take it down. Either way you
# don't want your demo to dead-end at "queued". This script keeps the bridge
# alive: if it exits for any reason other than Ctrl+C, restart with a short
# back-off so we don't tight-loop on a real crash.
#
# Usage:
#   cd ~/Downloads/Nexus/tender-agent
#   source ~/Desktop/tender-agent/.venv/bin/activate
#   bash scripts/run_bridge.sh
#
# Stop: Ctrl+C twice — once to kill the bridge, again to exit the wrapper.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG="${BRIDGE_LOG:-/tmp/bridge.log}"
BACKOFF_SECS=3

cd "$REPO_DIR"

# Trap Ctrl+C: tell the user we're shutting down deliberately, then exit clean.
intentional_exit=0
trap 'intentional_exit=1; echo ""; echo "[wrapper] Ctrl+C received, stopping."; exit 0' INT

while true; do
  echo "[wrapper] Starting bridge at $(date '+%H:%M:%S') — logs streaming to $LOG"
  python scripts/nexus_bridge.py 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$intentional_exit" = "1" ]; then
    break
  fi
  echo ""
  echo "[wrapper] Bridge exited with code $rc at $(date '+%H:%M:%S')."
  if [ "$rc" = "137" ]; then
    echo "[wrapper] Exit 137 = SIGKILL (likely macOS OOM killer or external kill -9)."
  elif [ "$rc" = "143" ]; then
    echo "[wrapper] Exit 143 = SIGTERM (something asked it to stop)."
  fi
  echo "[wrapper] Restarting in ${BACKOFF_SECS}s. Hit Ctrl+C to stop the wrapper."
  sleep "$BACKOFF_SECS"
done

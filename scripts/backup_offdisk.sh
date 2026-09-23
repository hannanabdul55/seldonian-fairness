#!/bin/bash
# Off-disk backup of what git does not protect, so that losing the WSL disk (as on
# 2026-09-19, see reports/recovered/README.md) no longer loses work.
#
#   - a git bundle of every ref, which includes commits not yet pushed;
#   - the working tree with untracked and ignored files (results/, logs/, spike
#     outputs), without .venv and caches, which can be regenerated;
#   - this project's Claude Code memory and session transcripts.
#
# Destination: $BACKUP_DIR (default D:\backups\seldonian-fairness, outside the WSL
# disk image). Nothing is ever deleted from the tree mirror (no rsync --delete),
# so a file deleted by accident here is still in the backup. Bundles are dated;
# the newest 14 are kept. Also warns when the WSL root is nearly full, because a
# full root going read-only was the start of the 2026-09-18/19 loss.
#
#   scripts/backup_offdisk.sh            # run once
#   crontab: */30 * * * * /home/hannanabdul/seldonian-fairness/scripts/backup_offdisk.sh
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${BACKUP_DIR:-/mnt/d/backups/seldonian-fairness}"
CLAUDE_PROJ="$HOME/.claude/projects/-home-hannanabdul-seldonian-fairness"
LOG="$DEST/backup.log"
MIN_ROOT_GB=5

mkdir -p "$DEST/tree" "$DEST/bundles" "$DEST/claude" || { echo "cannot write $DEST" >&2; exit 1; }
exec >>"$LOG" 2>&1
# one backup at a time
exec 9>"$DEST/.lock"
flock -n 9 || { echo "$(date -Is) another backup is running; skipped"; exit 0; }

echo "== $(date -Is) backup start"
status=0

free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "${free_gb:-0}" -lt "$MIN_ROOT_GB" ]; then
    echo "WARNING: WSL root has only ${free_gb} GB free (< ${MIN_ROOT_GB} GB)"
    command -v wall >/dev/null && echo "seldonian-fairness backup: WSL root has only ${free_gb} GB free" | wall 2>/dev/null
fi

# a new bundle only when some ref moved (the bundle is ~200 MB)
refs=$(git -C "$REPO" for-each-ref --format='%(objectname) %(refname)' | sha1sum | cut -c1-40)
if [ "$refs" != "$(cat "$DEST/bundles/.refs" 2>/dev/null)" ]; then
    stamp=$(date +%Y%m%d-%H%M)
    if git -C "$REPO" bundle create "$DEST/bundles/repo-$stamp.bundle" --all 2>/dev/null; then
        echo "$refs" > "$DEST/bundles/.refs"
        ls -1t "$DEST"/bundles/repo-*.bundle | tail -n +15 | xargs -r rm -f
    else
        echo "ERROR: git bundle failed"; status=1
    fi
fi

rsync -a --no-perms --no-owner --no-group \
    --exclude='.venv/' --exclude='__pycache__/' --exclude='.git/' \
    --exclude='.pytest_cache/' --exclude='.ruff_cache/' --exclude='*.pyc' \
    "$REPO/" "$DEST/tree/" || { echo "ERROR: tree rsync failed"; status=1; }

if [ -d "$CLAUDE_PROJ" ]; then
    rsync -a --no-perms --no-owner --no-group "$CLAUDE_PROJ/" "$DEST/claude/" \
        || { echo "ERROR: claude rsync failed"; status=1; }
fi

echo "== $(date -Is) backup done (status $status, root free ${free_gb} GB)"
exit $status

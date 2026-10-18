#!/usr/bin/env bash
# Cron: (re)launch the recon driver in tmux when work remains; flag low disk. Never emails itself.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA=$ROOT/data
OUTBOX=$DATA/recon_outbox.txt
note() { echo "$(date '+%F %T')  $*" | tee -a "$OUTBOX"; }

free_gb=$(df -BG --output=avail "$DATA" | tail -1 | tr -dc 0-9)
if (( free_gb < 20 )) && [[ ! -e $DATA/.recon_disk_warned_$(date +%F) ]]; then
  touch "$DATA/.recon_disk_warned_$(date +%F)"
  note "LOW DISK: ${free_gb} GB free on SW"
fi

[[ -e $DATA/recon_finished ]] && exit 0
tmux has-session -t recon 2>/dev/null && exit 0

# Stop relaunching a driver that keeps dying.
RELAUNCHES=$DATA/.recon_relaunches
now=$(date +%s)
recent=$(awk -v t=$((now - 7200)) '$1 > t' "$RELAUNCHES" 2>/dev/null | wc -l)
if (( recent >= 3 )); then
  [[ -e $DATA/.recon_crashloop ]] || { touch "$DATA/.recon_crashloop"; note "DRIVER CRASH-LOOPING: not relaunching; see data/recon_queue.log"; }
  exit 0
fi
echo "$now" >> "$RELAUNCHES"

[[ -e $DATA/recon_queue.log ]] && note "driver was not running; relaunched" || note "driver started"
tmux new-session -d -s recon "bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate nerfly \
  && cd $ROOT && python scripts/colmap/recon_queue.py >> $DATA/recon_queue.log 2>&1'"

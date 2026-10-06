#!/usr/bin/env bash
# Multi-view LP PILOT (cam1 + cam4), FOREGROUND. Launch from Windows with a hidden, detached
#   Start-Process wsl.exe -WindowStyle Hidden -ArgumentList '-e','bash','/mnt/c/Users/SabatiniLab/train_lp_multiview_fg.sh','<a|b>','<smoke|full>'
# (`wsl -e bash script` with nohup'd children dies when wsl returns -- 2026-10-05 pitfall.)
#   arm a = no calibration; arm b = 09-11 calibration (cam1+cam4) + 3-D augmentation + low-weight projection losses
#   smoke = 2 epochs, backbone unfrozen from epoch 1 (worst-case memory), patch masking exercised; full = the config
# GPU guard: kills ONLY this job's process group if the card passes 7000 MiB. Predict settings: DALI 16 frames.
set -u
ARM=${1:-a}; MODE=${2:-smoke}
source /root/miniforge3/etc/profile.d/conda.sh; conda activate lp
SRC=/mnt/c/Users/SabatiniLab/lp_stage/multiview-pilot-20261006
SRCC=/mnt/c/Users/SabatiniLab/lp_stage/multiview-pilot-20261006-calib
REPO=/mnt/c/Users/SabatiniLab/Github/widefield_pipeline
D=/root/lp/multiview-pilot-20261006
DC=/root/lp/multiview-pilot-20261006-calib
LOG=/mnt/c/Users/SabatiniLab/lp_stage/lp_train_mv_${ARM}_${MODE}.log
STAMP=$(date +%Y%m%d)
echo "== $(date) multiview pilot arm=$ARM mode=$MODE" > "$LOG"

# data on ext4 (copied once; the Windows staging copy stays the source)
if [ ! -f "$D/CollectedData_cam1.csv" ]; then
  mkdir -p "$D" && cp -r "$SRC/labeled-data" "$SRC/videos" "$SRC"/CollectedData_cam*.csv "$SRC/moments.csv" "$SRC/export_report.json" "$D/"
  echo "copied data -> $D" >> "$LOG"
fi
if [ ! -f "$DC/calibration.toml" ]; then
  mkdir -p "$DC"
  for f in labeled-data videos CollectedData_cam1.csv CollectedData_cam4.csv; do ln -sfn "$D/$f" "$DC/$f"; done
  cp "$SRCC/calibration.toml" "$SRCC/calibration_SOURCE.txt" "$DC/"
  echo "built $DC (symlinks + calibration.toml)" >> "$LOG"
fi
if [ -e "$D/calibration.toml" ] || [ -d "$D/calibrations" ]; then
  echo "ABORT: $D must not hold a calibration (LP would silently make arm a calibrated)" >> "$LOG"; exit 2
fi
cp "$REPO/configs/lightning_pose_multiview_pilot.yaml" "$D/config.yaml"

OV=()
if [ "$ARM" = "b" ]; then
  OV+=(data.data_dir=$DC data.video_dir=$DC/videos eval.test_videos_directory=$DC/videos training.imgaug_3d=true
       +losses.supervised_pairwise_projections.log_weight=4.0
       +losses.supervised_reprojection_heatmap_mse.log_weight=3.0)
fi
if [ "$MODE" = "smoke" ]; then
  OV+=(training.max_epochs=2 training.min_epochs=1 training.check_val_every_n_epoch=1 training.unfreezing_epoch=1
       "training.lr_scheduler_params.multisteplr.milestones=[1]"
       training.patch_mask.init_epoch=0 training.patch_mask.final_epoch=2 training.log_every_n_steps=5)
  OUT=/root/lp/mv_smoke/arm${ARM}_${STAMP}
else
  OUT=$D/models/mv_pilot_arm${ARM}_${STAMP}
fi
echo "output -> $OUT" >> "$LOG"
echo "overrides: ${OV[*]}" >> "$LOG"

setsid litpose train "$D/config.yaml" --output_dir "$OUT" --overrides "${OV[@]}" >> "$LOG" 2>&1 &
PID=$!
PEAK=0
while kill -0 $PID 2>/dev/null; do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$U" -gt "$PEAK" ] && PEAK=$U && echo "GPU peak so far: $PEAK MiB ($(date +%T))" >> "$LOG.mem"
  if [ "$U" -gt 7000 ]; then echo "GUARD: $U MiB -> kill process group $PID" >> "$LOG"; kill -- -$PID; sleep 5; kill -9 -- -$PID 2>/dev/null; fi
  sleep 2
done
wait $PID; RC=$?
echo "== $(date) exit $RC; GPU peak $PEAK MiB" >> "$LOG"

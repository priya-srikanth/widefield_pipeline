#!/usr/bin/env bash
# Run the slide-21 variants ONE AT A TIME; each waits for free RAM >= 8 GB; after each, append its readout row.
# A variant that fails is retried once. Usage: chain.sh A_pre D_scale ...
S=C:/Users/SABATI~1/AppData/Local/Temp/claude/179e1046909c5be2/782a4a7a-7855-40ed-b449-494e8aa4b6bc/scratchpad
D=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/decomp_slide21_20261117
REPO=C:/Users/SabatiniLab/Github/wf_cleanup2
PY=C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe
free() { powershell -NoProfile -c '[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)'; }
for V in "$@"; do
  for try in 1 2; do
    until [ "$(free)" -ge 8 ]; do echo "$(date '+%T') $V waiting for RAM ($(free) GB)"; sleep 120; done
    echo "$(date '+%F %T') START $V try $try (free $(free) GB)"
    bash $S/run_variant.sh "$V"; rc=$?
    echo "$(date '+%F %T') END $V try $try rc $rc"
    [ $rc -eq 0 ] && break
    cp $S/runs/$V.log $S/runs/$V.try$try.log
  done
  [ $rc -ne 0 ] && { echo "FAILED $V twice"; continue; }
  TAG=$(grep -m1 "^# variant" $S/runs/$V.log | sed 's/.* tag //')
  (cd $REPO && PYTHONIOENCODING=utf-8 $PY docs/status/deck_swap_20261117/slide21_readout_table.py \
      "$D/$V/grant_figures/$TAG/epoch/data" "$V") 2>&1 | tee $S/runs/$V.readout.txt
done
echo "$(date '+%F %T') CHAIN DONE"

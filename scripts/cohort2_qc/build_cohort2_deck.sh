#!/usr/bin/env bash
# Rebuild the cohort2 preprocessing decks WITH the epileptiform-discharge + 490-LED QC/clip slides.
#
# preprocess_deck.py builds the bare per-animal motion-QC + photobleach decks (cohort-aware; it SPLITS
# cohort2 by indicator -> _PS118-PS120 Thy1 and _PS121-PS122 RiboL1). This wrapper then regenerates the
# cohort2 QC figures and appends the discharge screen, the 490-vs-470 panel, and the per-discharge
# trace+map panels + the cam4 (face) / slow-mo (map) video clips to the RiboL1 deck.
#
# Run in the `wfield` env on a box that can read N: MICROSCOPE. Override PY / WORK for another machine.
# Add a build_event_traces line per new PS122 discharge session as they accumulate.
set -e
PY="${PY:-C:/ProgramData/anaconda3/envs/wfield/python.exe}"
export WORK="${WORK:-E:/labcams_data}"                # scratch for regenerated figures (read by the QC scripts)
LAB="${LAB:-N:/MICROSCOPE/Priya/Widefield/labcams}"   # analysis desktop: LAB=M:/MICROSCOPE/Priya/Widefield/labcams
RAW="${RAW:-E:/labcams_data}"                         # raw .dat (imaging box only; not archived to the share)
RIBO="$LAB/cohort2/cross-session_preprocessing_PS121-PS122.pptx"
SZ="$LAB/cohort2/_tmp_qc_figures_20261031/seizure"    # pre-made cam4/slow-mo clips live here
cd "$(dirname "$0")/../.."                             # repo root

# 1. cohort2 QC figures (screen every PS121/PS122 date; 490-vs-470 SNR + droop)
"$PY" scripts/cohort2_qc/discharge_screen.py          # -> $WORK/cohort2_discharge_screen.png + .csv
"$PY" scripts/cohort2_qc/led_490_vs_470.py            # -> $WORK/cohort2_490_vs_470.png
"$PY" scripts/cohort2_qc/led_490_vs_470_dprime.py     # -> $WORK/cohort2_490_vs_470_dprime.png (response amp / SD / d')

# 2. per-discharge trace + dF/F-map-propagation panels (one line per confirmed PS122 discharge session)
"$PY" scripts/cohort2_qc/build_event_traces.py 20261104 PS122_490_20261104_150738 "$WORK/PS122_1104_event_traces.png"
"$PY" scripts/cohort2_qc/build_event_traces.py 20261029 PS122_20261029_192217     "$WORK/PS122_1029_event_traces.png"

# 2b. per-session per-spout cue / first-lick maps (native pixels) -> labcams/cohort2/spout_maps_native/, placed by
#     preprocess_deck as the `spout_native` slide type (Priya 11-05). Discharge trials are dropped inside.
"$PY" scripts/cohort2_qc/spout_maps.py all --per-session

# 2c. 11-04 photobleach figures re-drawn with the 490 nm legend (PS121 / PS122 only; PS118 stayed 470). Needs the
#     raw .dat, which only the imaging box has (only the camlog is archived). Guarded: without the raw,
#     photobleach.run's merge would DROP the existing 11-04 records whose raw it cannot see and rewrite the summary.
if [ -f "$RAW/20261104/PS121_490_20261104_122657/raw_widefield_data/pco_edge_run000_00000000_2_509_480_uint16.dat" ]; then
  WIDEFIELD_COHORT=cohort2 "$PY" -m wfield_local.preprocess 20261104 --only PS121 PS122 --skip-preprocess \
    --skip-maps --skip-xall --skip-crossday-intensity
else
  echo "11-04 raw .dat not under $RAW: photobleach PNGs keep their old 470 legend (re-run this on the imaging box)"
fi

# 2d. PS122 11-04 discharge clips (490 map | cam4 face | traces; the largest discharge 5x slowed) + traces -> $SZ
"$PY" scripts/cohort2_qc/seizure_events_video.py --day 1104

# 3. base preprocessing decks (writes _PS118-PS120 and _PS121-PS122 beside this output name)
WIDEFIELD_COHORT=cohort2 "$PY" -m wfield_local.preprocess_deck --output "$LAB/cohort2/cross-session_preprocessing.pptx"

# 4. append QC summaries + discharge clips to the RiboL1 deck (append_qc_slides embeds mp4 with a poster)
"$PY" scripts/cohort2_qc/append_qc_slides.py "$RIBO" \
  "Epileptiform-discharge screen - PS121/PS122 all dates (PS122 recurrent 10-29 and 11-04)::$WORK/cohort2_discharge_screen.png" \
  "490nm vs 470nm LED - whole-signal SD / shot floor (higher at 490) + within-session droop (no excess)::$WORK/cohort2_490_vs_470.png" \
  "490nm vs 470nm LED - task response: dF/F and trial SD both up at 490, single-trial d' = best 470 day::$WORK/cohort2_490_vs_470_dprime.png" \
  "PS122 epileptiform discharge 11-04 (490nm) - traces + dF/F map propagation::$WORK/PS122_1104_event_traces.png" \
  "PS122 epileptiform discharge 10-29 - traces + dF/F map propagation::$WORK/PS122_1029_event_traces.png" \
  "PS122 11-04 discharges (490nm), 75.2-78.2 min - face/behavior video (cam4) + map + traces::$SZ/PS122_1104_events_cam4.mp4" \
  "PS122 11-04 largest discharge (65.4 min, 490nm) - widefield dF/F map slow-motion::$SZ/PS122_1104_event_slowmo.mp4" \
  "PS122 10-29 discharge - face/behavior video (cam4, H.264)::$SZ/PS122_1029_events_cam4.mp4" \
  "PS122 10-29 discharge - widefield dF/F map slow-motion::$SZ/PS122_1029_event_slowmo.mp4" \
  "PS122 10-30 - event traces (sub-threshold, vascular)::$SZ/PS122_1030_event_traces.png"
echo "cohort2 decks rebuilt with discharge + 490 QC/clips -> $LAB/cohort2/"

# Figure provenance re-run, 2026-11-16 (a8)

`FIGURE_PROVENANCE_20261116_rerun.csv` = window 34's 11-06 table (`FIGURE_PROVENANCE_20261106.csv`, kept unchanged)
re-generated with 34's own scripts (`prov/parse_steps.py` -> `look3.py` -> `merge.py`, copied read-only from 34's
session scratchpad and run in a copy) against the render logs as they stand on 11-16. Nothing else changed: same
replacement-pass map, same per-script agent tables, same captions (11-06), same status rules (section 5 of the 11-06 md:
CONFIRMED = every mapped step done, env read from its log, window / licks / trials / stats from the code).

Result: CONFIRMED 45 -> 257 of 832 (INFERRED 635 -> 423; UNKNOWN 152 unchanged). PI-deck rows: CONFIRMED 8 -> 38,
INFERRED 108 -> 78, UNKNOWN 71. Render steps of queues 7b-8g: 247 done, 156 pending (no log under that step name;
some were re-defined in later queues under other names), 16 failed, 7 skipped, 6 running.

Still not confirmable this way (deck rows): no producing script named 48; figure not a `_daqlk` render step (older
kept runs, code defaults only) 47; a key field missing 20; step pending / failed / skipped in 7b-8g ~30.

Caveat: CONFIRMED describes the figure's current producing step. A deck image cropped before that step re-rendered
(the 11-02 build inputs) can predate it; that is not checked here.

Deck notes "Data:" blocks written from these rows: `docs/status/DECK_DATA_BLOCK_PASS2_2026-11-16.csv` (26 slides);
first pass `DECK_DATA_BLOCK_PASS_2026-11-16.csv` (21 slides). Deck: 58 / 206 slides carry the block.

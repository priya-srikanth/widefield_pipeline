# PI deck swap pass, 2026-11-17 (a8)

Priya 11-17, answering the per-figure provenance sweep (DECK_STALE_FIGURES_2026-11-16.csv):
1. group A + B (74 slides): "swap in rebuilt figures and rewrite slide and notes text"
2. group C (35 slides): "run them" (new renders on the rebuild)
3. held slides (slide 13, balance 135-140 / 203): replace with the updated versions where they exist.

Deck: `M:/MICROSCOPE/Priya/Widefield/presentations/Overview_2026-11-06.pptx` (206 slides).
Backup before this pass: `Overview_2026-11-06.bak_pre_a8_swap_20261117.pptx` (same folder).
Only a8 writes the deck, serially, from the per-slide specs below. Helpers NEVER open the deck for writing.

Inputs for helpers:
- `stale_slides_dump.json` (this folder): per stale slide = its stale-table row, hidden flag, every shape (name, box in
  EMU, picture pixel size + crop fractions, text paragraphs) and the notes paragraphs.
- `docs/status/DECK_STALE_FIGURES_2026-11-16.csv`, `docs/status/provenance_sweep_20261116/sweep_all.json`.
- Render logs: `C:/Users/SabatiniLab/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/render_daqlk/logs/<step>.log`
  (heads carry `# env` lines and the code sha); the rebuild provenance table
  `docs/status/rebuild_20261103/FIGURE_PROVENANCE_20261116_rerun.csv`.

## Per-slide spec (one JSON object per slide; a helper writes a list to `specs/<batch>.json`)

```json
{
  "slide": 21,
  "title_now": "first text of the title shape, to assert the slide has not moved",
  "status": "ready | needs_priya | skip",
  "skip_reason": "",
  "pictures": [
    {"shape_name": "Picture 5", "new_image": "M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-17/swap_inputs/s021_state_position.png",
     "source_render": "the rebuild render the image was cut from", "crop_done": "title cut, as the old image"}
  ],
  "slide_text": [{"shape_name": "TextBox 3", "old": "exact old paragraph text", "new": "new paragraph text"}],
  "notes_body": ["new notes paragraphs AFTER the Data: block (full replacement of the old body)"],
  "data_block": {"Window": "", "Licks": "", "Trial period": "", "Trial types": "", "Gate": "", "Baseline": "",
                 "Correction": "... [traced: render log <step>.log]"},
  "changes": [{"what": "position chronic change vs pre", "old": "-.09 **", "new": "-.077, Holm .057 n.s.",
               "source": "<step>.log line N"}],
  "conclusion_changed": true,
  "conclusion_note": "one sentence for Priya if the slide's main point changes"
}
```

Rules:
- New image = the CURRENT rebuild render (tag `..._daqlk`), with the same crop as the old image (title cut, suptitle
  removed, re-tiled ...). Write it under `M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-17/swap_inputs/`
  as `sNNN_<short>.png`. Never overwrite an existing file; never write outside MICROSCOPE/Priya or your scratchpad.
- Every number on the slide and in the notes comes from a render log / output table of the rebuild; cite it in
  `changes`. If a number cannot be traced, keep the sentence out and say so in `changes` (old -> "removed: untraced").
- If the main point (the title claim) no longer holds, set `conclusion_changed` and rewrite the title too (in
  `slide_text`); status `needs_priya` only when the right rewrite is a scientific judgment you cannot make from the logs.
- Captions: replace, don't annotate (no "CORRECTION:" / "updated" notes). American spelling. Stars = boot_delta CI
  (LMM only as a dagger backup). Every readout = change from the animal's own pre. Notes keep the methods; the slide
  keeps the main point. Name the actual statistic (max-statistic, not "Bonferroni", where the code uses maxstat).
- Data: block = the 7 labels exactly; the Correction line ends with `[traced: render log <step>.log]` or
  `[traced: code + output file, no run log]`.
- Kill only processes you started (exact PIDs). The box is shared: at most 4 worker processes per render.

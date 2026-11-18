# grants/ — code that made grant figures

One folder per grant submission. The code here composes **figures for grant applications** from outputs the
pipeline has already written (session / LMM / bootstrap CSVs, epoch SVGs, deck PNGs, Illustrator panels).

**Nothing in this folder is part of the analysis pipeline.** Nothing here recomputes a result, nothing in
`wfield_local/` imports it, and the nightly does not run it. If a number on a grant figure is wrong, fix it in the
pipeline and re-pull, not here. That is also why `grants/` sits outside `scripts/` and `wfield_local/`: the
pipeline guards in `tests/` (figure-layout writers, library-imports-scripts, CLI checks) scan those trees and do not
apply to one-off figure composition.

Not to be confused with **`wfield_local/grant_figures.py`** (and `epoch_grant_figures.py`, `grant_kit.py`, ...):
that is the *pipeline's* summary figure set, rendered every night into `grant_figures/` on the server. The code here
reads from that set; it does not replace it.

| Folder | Grant | What it holds |
|---|---|---|
| [`k08_2026/`](k08_2026/README.md) | NIH K08 resubmission, November 2026 | Panel renderers, figure compositors and SVG / PDF post-processing for Research Strategy Figures 3, 4 and 5 |

## Adding a grant

Make `grants/<grant>_<year>/` with a `README.md` saying which figures it builds, where the inputs came from, and
how to re-run it, and add a row to the table above and to branch N of [`docs/START_HERE.md`](../docs/START_HERE.md).
Keep the grant's own documents (drafts, `.docx`, `.ai`, rendered figures) in the grant folder on the laptop, not
here: this repo is public.

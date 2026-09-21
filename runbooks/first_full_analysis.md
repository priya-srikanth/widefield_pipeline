# Runbook — a first full analysis, from raw sessions to the deck

**Who this is for:** someone who has never run this pipeline and wants one animal's data to come
out the far end as figures they can read. The other three runbooks assume you already know the
shape of the night; this one does not.

**What it is not:** a script to paste. Every stage here takes between minutes and hours, runs on
one of two or three machines, and several will refuse to proceed if an earlier one did not finish.
The refusals are the point — see *When something refuses* below.

Before starting, read [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for what the stages are.
Nightly operation is [`imaging_computer_nightly.md`](imaging_computer_nightly.md) and
[`analysis_computer_nightly.md`](analysis_computer_nightly.md); this walkthrough calls the same
entry points those do, one at a time, so you can see what each produces.

---

## 0 · Ground rules you cannot work around

Four rules are enforced in code, not by convention. Knowing them saves an hour of confusion the
first time one fires.

1. **Never delete original data; only ever write inside `MICROSCOPE/Priya/…`.**
   [`writeguard`](../wfield_local/writeguard.py) raises on a write outside that subtree. Tests are
   additionally forbidden from writing to any share at all — a one-byte file from a test helper
   once re-created a whole session directory on the live share.
2. **Raw movies live on standby, not MICROSCOPE.** `.dat` and `.bin` go to `M:` via
   [`archive_day`](../wfield_local/archive_day.py). If you are looking for a raw movie on `N:` and
   it is not there, that is correct.
3. **Per-session loops fan out over cores.** Use [`parallel`](../wfield_local/parallel.py) /
   `analysis_kit.fan_sessions`, collect in sorted order, and pass options in the ITEM — a spawned
   worker does not inherit module globals.
4. **Any number in a summary table must be reconstructible from a written CSV.** A figure whose
   values exist nowhere else drifts from every text that quotes it and nothing can notice.

Which machine you are on decides which paths resolve. Check first:

```bash
python -c "from wfield_local.paths import PathResolver; r=PathResolver(); print(r.machine); print(r.root('labcams'))"
```

---

## 1 · Preprocessing — imaging box, hours

Raw is on local `E:`. This discovers the date's sessions, motion-corrects, and writes per-session
output under `labcams/<date>/<session>/`.

```bash
python -m wfield_local.archive_day upload-daq 20260818
```

```bash
python -m wfield_local.nightly 20260818 --machine imaging
```

`nightly` sequences `preprocess` → `preprocess_deck` → `archive_day archive` + `verify`. To watch
one stage on its own, run `python -m wfield_local.preprocess 20260818` and read
[`imaging_computer_nightly.md`](imaging_computer_nightly.md) for the per-stage flags.

**What to check before moving on:** `frames_average` should have a mean in the low tens of
thousands. A value several times smaller means a partially-written transfer — the file will have
the *correct byte count* and be zeros past the cut, which is why size and mtime cannot detect it.
That failure has happened and cost 648 GB of quarantine.

## 2 · Decomposition — GPU box, hours

LocaNMF is not driven by a bare date. [`run_locanmf`](../wfield_local/run_locanmf.py) fits ONE
session and wants explicit paths (`--allen-dir`, `--output`, `--label`);
[`batch_locanmf`](../wfield_local/batch_locanmf.py) fits a list of them from a manifest:

```bash
python -m wfield_local.batch_locanmf --manifest <sessions.json>
```

The manifest is a JSON list of `{allen_dir, label, output}`. `--mode` selects `locanmf`, `snmf` or
both; the r2 / locality / maxrank thresholds default from `defaults.yaml` rather than being
retyped.

[`joint_locanmf`](../wfield_local/joint_locanmf.py) is a **library, not a CLI** — `build(animal,
sessions)`. It fits one basis across an animal's sessions and then FREEZES it, so later days are
projected onto it rather than refitted. That is what makes a cross-day comparison a claim about
cortex instead of a claim about the parcellation. A missing frozen basis is reported and skipped,
**never silently refitted**.

## 3 · Analysis and figures — analysis box

In practice you do not run stage 3 by hand on a first pass. `await_locanmf` polls for the night's
LocaNMF output, registers the sessions when it lands, and then runs the whole figure sequence:

```bash
python -m wfield_local.await_locanmf 20260818 --skip-grant
```

`--once` does a single detection pass instead of polling; `--no-locanmf` assumes it has already
run; `--skip-grant` drops the two-hour render, which is the flag you want the first time.

If LocaNMF is already registered, go straight to the figures:

```bash
python -m wfield_local.nightly 20260818 --machine analysis --from 0606-0918
```

Either route ends in [`nightly_figs`](../wfield_local/nightly_figs.py), which is the real sequence:
the per-day decoders and encoders, then the cross-session frozen arms, then
`poststroke_section_g`, `position_coding_directions`, `grant_figures`, `epoch_grant_figures`, and
finally the deck.

**The grant render is the long pole**: about two hours from a cold bootstrap cache, minutes from a
warm one. `--skip-grant` exists for the nights where it does not fit. To inherit a warm cache from
another box:

```bash
set WIDEFIELD_SESSION_CACHE=N:/MICROSCOPE/Priya/Widefield/session_cache_v12
```

To render one figure family while you are finding your feet:

```bash
python -m wfield_local.grant_figures --only 1 1b --jobs 1 --output ./scratch_figs
```

**Always pass `--output` to somewhere local when experimenting.** The default writes the published
deliverable.

## 4 · The deck

```bash
python -m wfield_local.locanmf_analysis_deck --src <figures_working> --out ./scratch_deck.pptx
```

531 slides at last count. The interesting part is not the build but the gates — see below.

---

## When something refuses

**Most of the ways this pipeline stops are deliberate, and the message names the fix.** The
refusals exist because each of them once did not, and a bad result was published instead.

| refusal | what it means |
|---|---|
| `DeckIncomplete` | the rebuild could not find every figure, and there is already a published deck. An upstream step failed quietly. Fix it, or `--allow-missing N`. |
| `DeckUnresolved` | notes quote sidecar values this build could not find, so the slides would read `[[? … sidecar missing]]` instead of numbers. **Usually a stale checkout after the figure tree moved — pull first.** |
| `DeckFromFailedRun` | a nightly step failed earlier in the same run; the deck will not publish on top of it. |
| a `writeguard` error | you are writing outside Priya's subtree, or a test is touching a real share. |
| `assert_writable` on the output dir | same, for a figure renderer's `--output`. |

A deck with holes in it is a perfectly valid deck, which is precisely why these gates are code and
not judgement.

---

## Verifying you have not broken anything

If you change code rather than just run it, the suite is not enough — it does not cover the
analysis scripts, deliberately. **Numbers are the test.**

```bash
python -m pytest tests/ -q
```

```bash
python -m wfield_local.grant_figures --jobs 8 --output ./before
```

…make the change, render to `./after`, and byte-compare the PNGs. For SVGs use
[`scripts/compare_svg_renders.py`](../scripts/compare_svg_renders.py) — matplotlib writes a
timestamp and random element ids into every SVG, so a plain byte comparison calls all of them
different whatever you did. For the deck use
[`scripts/deck_fingerprint.py`](../scripts/deck_fingerprint.py).

Two rules that have each caught a real bug:

- **Run the baseline twice first.** A baseline you have not shown to be reproducible is not a
  baseline.
- **Check that your check can fail.** Introduce the defect deliberately and confirm the comparison
  notices. Three tools in this repo have at some point reported success having measured nothing.

---

## Where to look next

| | |
|---|---|
| why any of this is the way it is | [`DECISIONS.md`](../DECISIONS.md) |
| what the modules are and how they relate | [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) |
| what was true on a given day | [`docs/status/`](../docs/status) |
| the rules, in force | [`CLAUDE.md`](../CLAUDE.md) |

# Cortical results pages renamed and reorganized (Priya 2026-11-18)

| page | URL | old names | new `<title>` | version |
|---|---|---|---|---|
| target | https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV | "Cortical Spout Coding After Stroke", "the imaging page" | **Spout (Target) Coding After Stroke** | v62 -> **v63** |
| movement | https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d | "Attempt Encoding Figures", "the goal (question) page", "the encoding page" | **Movement (Execution) Coding After Stroke** | v322 -> **v323** |

Built on the live versions (Artifact read 11-18: target version 1795015877-6281, movement 1795015841-cf7b) with
`edit_pages.py` in this folder (every replacement asserted to hit exactly once). No figure, caption number or section /
figure number changed.

## What changed
- **Target page:** `<title>` and h1 ("Spout (target) coding after a striatal stroke: where the cortical spout code goes");
  a `#pagerole` line under the h1 says what the page covers and links the movement page. The scope sentence now sends
  kinematic regressors and quiet events to the movement page. Section 15 (quiet-period events) is reduced to a pointer
  paragraph linking movement-page section 23 (`#quiet-state`); the section id `quiet` and all seven figure ids
  (`f-q_fr`, `f-q_examples`, `f-q_fr_detail`, `f-q_threshold`, `f-q_spectrum`, `f-q_sdmaps`, `f-q_eta_maps`) are kept
  as empty anchors so old links land; the status-table row keeps its 15.x links. Cross-references added to sections 5
  (-> movement section 8, `#decoder`) and 8 (-> movement section 9, `#maps`). Old-name mentions fixed: status row
  ("movement page section 23"), Fig 5.1 / 5.2 caption source notes ("movement page" / "movement page script"),
  section 14 Haufe note, Pending list.
- **Movement page:** `<title>` and h1 ("What cortex carries about the executed movement after a striatal stroke: lick,
  tongue, face and state"); `#pagerole` line linking the target page. Cross-references added to sections 8 (-> target
  section 5, `#attempts`) and 9 (-> target section 8, `#attmaps`). Old-name mentions fixed: status bullet "(imaging
  page, FINDINGS 3.12)" -> link to target section 6; section 23 tag and intro ("moved from the spout (target) page").

## Source copies
Neither candidate source matched the live target page: `docs/results_pages/imaging/imaging_page/imaging_results.html`
(repo, = v46, ~420 differing lines) and `M:/MICROSCOPE/Priya/Widefield/imaging_page_src/imaging_page/imaging_results.html`
(11-01). `_goal_page/index.html` (repo) is the v311 movement page, not v322. None was overwritten; the live pages are
the source (build any republish from an Artifact read). `docs/results_pages/imaging/README.md` now says so.

## Repo text updated
START_HERE (new "two cortical results pages" paragraph + branches F / G / H), FINDINGS "Results pages" line,
ENCODING_FINDINGS section 8 header, CUT_FIGURES header, templates/README, results_pages/imaging/README; deck code page
labels (`deck_keyfigs.KEY_FIGURES_PAGES` -> "Spout (Target) Coding", `deck_registry._GOAL_PAGE` -> "Movement
(Execution) Coding", the KEY FIGURES divider text in `locanmf_analysis_deck.py`, `tests/test_deck_keyfigs.py`).
Dated status / handoff docs keep the old names (historical).

## PI deck (not edited here; another window owns `Overview_2026-11-06.pptx`)
`deck_mentions_Overview_2026-11-06.txt` lists every slide whose notes (or slide text) name the old pages, from a
read-only copy of the deck as of 11-18 10:37 (slide number, h = hidden). Slide text: slides 2 and 10 ("Cortical spout
coding" as the agenda / divider name). Notes: "imaging page" on 4, 11, 13, 15, 18, 21-27, 188-190, 206-208; "encoding
page" / "goal page" on 3, 8, 9, 64-71, 99-104, 106, 114, 118-159 (many), 195-204.

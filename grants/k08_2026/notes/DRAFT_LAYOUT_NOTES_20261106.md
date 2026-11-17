# K08 figure drafts — how these are built and how to edit them

## What this is
Three "everything in" layout drafts, built to match the MGN progress-report `.ai` conventions:
**Arial throughout**, panel letters **Arial Bold 11 pt**, figure width **7.4 in** (the full Research
Strategy text width). Artboard width is US Letter; artboard *height* grows to fit, so Figure 1 is
currently about 25 in tall. That is deliberate: these hold every panel selected on 6 Nov so they can
be compared and pared down, not shipped as is.

| File | Panels | Status |
|---|---|---|
| `Figure1_model_and_deficit_DRAFT.svg` | a–m (13) | everything in; two placeholders |
| `Figure2_representation_DRAFT.svg` | a–d (4) | panel c and d still need swapping/subsetting |
| `Figure3_facerhythm_encoder_DRAFT.svg` | a–g (7) | everything in |

## Opening and editing
Open the `.svg` directly in Illustrator. Vector panels stay **live and editable** — source SVGs are
inlined, not linked or rasterised, so text, axes and data marks are real objects. Deck-derived
panels are embedded PNGs and are not editable beyond scaling and cropping.

Each panel is its own named group (`Panel_a`, `Panel_b`, ...) plus a red `NOTES` group to delete
before use. Every id inside each panel was namespaced on import (`a_`, `b_`, ...) so clip paths and
glyph definitions from different sources cannot collide. Do the same if you add a panel by hand, or
expect stray clipping.

## Figure 1 panel map
a VLS histology **(placeholder — reuse existing grant panel)** · b fluoroscopy swallow frames ·
c inter-swallow interval · d licks per interval · e swallow duration · f sitting transit ·
g bolus density · h six-spout schematic **(placeholder — reuse existing grant panel)** ·
i hit rate by spout · j cam1+cam4 labelled frame, tongue out · k first-lick aim by spout ·
l 3-D paths across epochs · m far-contra trial outcomes

Order follows the agreed rule: fluoroscopy opens (deficit first, then the assay built to measure
it), and labelled frames come before the aim panel, which comes before the trajectories.

## Figure 3 panel map
a FaceRhythm master face · b the 12 frozen FR factors · c lick_right drops / lick_left substitutes ·
d FR loading-change maps · e encoder residual maps (lick + cue windows) · f cued ROI row ·
g per-ROI cued traces by epoch

## Known to-dos
**Figure 1** — f (sitting transit) is natively 772 x 189 pt and renders too small at this width;
enlarge or split. m (far-contra trial outcomes) is a dense grid and is illegible below about 5 in.
a and h are placeholder boxes.

**Figure 2** — panel c is the per-position delta, not the **pooled** refit − frozen scatter that was
asked for. Panel d is still the full 5 epochs x 6 positions grid; subset to the **far-contra column**.

**Figure 3** — e still needs the **pre-stroke residual level** alongside the difference maps.
f should be subset to **tjM1 and SSp-m, R and L only**. The cued FIRST-lick readout-offset panel and
the per-ROI far-contra-only panel have not been located yet.

**All three** — no captions yet, and panel heights are not balanced within rows.

## Rebuilding
`build_fig.py` does the inlining and id-namespacing. `make_figs.py` holds the Figure 1 and Figure 3
layouts as plain row lists — edit the fractions and panel order there and re-run. `make_fig2.py` is
the older per-figure script for Figure 2. `svg2pdf.py <file.svg>` renders a PDF and a full-page PNG
preview via headless Chromium.

## Evidence filter
Only **STANDS** results carry an argument. PROVISIONAL panels need the caption to say so. The
chronic jaw-steering deficit (NOT ESTABLISHED, 2/4 animals) and the acute 1–4 Hz laterality result
(WITHDRAWN, T22) must not appear in any of these figures.

"""Imaging page: add section 14, the Haufe-transformed decoder maps (Priya 10-27).

Nine figures from the epoch render (grant_figures/epoch/, rendered 10-21): family 14 (pooled maps + acute-pre test),
14z (each epoch against zero) and 14pa (far contra per animal), each in the pre-cue, cue and lick windows. Methods
traced from wfield_local/epoch_maps.py (_fig_14_beta_maps, _fig_14z_beta_vs_zero, _fig_14pa_beta_maps_by_animal) and
wfield_local/beta_maps.py (session_maps, maps_by_epoch, significance_contour, vs_zero_contour). Runs after
p_img_titles.py. PNGs are copied into imaging_page/ by page_figs.py (haufe_*, haufez_*, haufepa_*).
"""
from pathlib import Path

P = Path(__file__).parent / "imaging_page" / "imaging_results.html"
s = P.read_text(encoding="utf8")
assert 'id="haufe"' not in s, "already added"

EPO = ("Epochs per animal from its own lesion day (acute = far-contra hit rate &lt; 25% of pre: PS92 days 1&ndash;5, "
       "PS93 1&ndash;4, PS94 1&ndash;7, PS95 1; chronic from day 11 / 11 / 25 / 15; subacute between). Pre = all "
       "curated pre-stroke sessions (June + August), 44 sessions, 11 per animal.")
DEC = ("<b>Decoder:</b> REFIT in every session (not frozen): multinomial L2 logistic regression (C = 0.5, class-balanced "
       "weights, standardised features) on the session&rsquo;s rank-100 SVT temporal components, all time bins of the "
       "window as features; 5-fold block cross-validation, coefficients averaged over folds and time bins. "
       "<b>Haufe transform:</b> each position&rsquo;s weight vector b is turned into an activation pattern A = Cov(X)&nbsp;b "
       "(covariance over the training trials) and projected to pixels as U&nbsp;@&nbsp;A, so a pixel&rsquo;s value is "
       "cov(pixel, decoder output for that position) &mdash; where the signal is, not what the decoder multiplies "
       "(pattern vs raw weight r = .245). <b>Reference:</b> MEAN over all trials (one-vs-rest), so the six positions "
       "are COUPLED: red = more active on this position&rsquo;s trials than on the average trial; a position that loses "
       "drive flips sign and hands the other five an unearned increase. A position needs &ge; 20 trials in a session "
       "or that session&rsquo;s map is refused.")
WIN = {
    "precue": ("pre-cue (ENL): the 2 s ending at the cue, slid earlier if a lick falls inside it; working trials = "
               "licking trials plus misses while still working (terminal quit period excluded)"),
    "cue": ("0&ndash;2 s after the cue; working trials = licking trials plus misses while still working (terminal quit "
            "period excluded)"),
    "lick": "0&ndash;2 s after the first lick; lick trials only (licks from the DAQ spout-contact sensor)",
}
WTXT = {"precue": "pre-cue", "cue": "cue", "lick": "lick"}
POOL = ("<b>Pooling:</b> each animal&rsquo;s session mean divided by its own pre-stroke RMS (per position), then "
        "averaged over animals present in both epochs; difference columns = epoch &minus; pre within animal on that "
        "scale; one colour scale for the map columns (the figure&rsquo;s own subtitle &lsquo;PER ROW&rsquo; is stale), "
        "difference columns expand if larger. Panel titles give animals, sessions, trials and split-half r of that "
        "epoch&rsquo;s mean map.")
NEST = ("nested animals &rarr; sessions bootstrap (2,000) on 2,022 bins (8&times; downsampled, eroded mask without "
        "bulbs / glue), studentised, family-wise &alpha; = .05 by max |z|; a panel is suppressed when its significant "
        "bins are &gt; 2&times; concentrated in the rim")
STATS14 = (f"<b>Stats:</b> contours on the ACUTE &minus; PRE column only: {NEST}. Subacute and chronic differences are "
           "drawn, not tested. The printed acute amplitude ratios are RMS(acute) / RMS(pre) and count a sign flip as "
           "amplitude, so they are not a deficit measure under this reference.")
STATSZ = (f"<b>Stats:</b> each epoch tested against ZERO on its own: {NEST} (the figure&rsquo;s subtitle says "
          "&lsquo;Bonferroni&rsquo;; the code uses the max-statistic). Maps and test pre-normalised per animal as in "
          "Fig 14.1. Two panels are separate statements, never a difference.")
SCRIPT = "<b>Script:</b> <code>wfield_local/epoch_maps.py</code> (maps and tests <code>wfield_local/beta_maps.py</code>)."

FIGS = [
    # fid, png, kind, window, title, reading
    ("fig14-1", "haufe_precue", "14", "precue",
     "Haufe decoder maps, pre-cue window, pooled: weak and unreliable post-stroke; only near middle&rsquo;s acute change "
     "is significant",
     "Pre-cue maps are weak (pre split-half r .73&ndash;.96). Acute far contra turns broadly positive (more active than "
     "the average trial) and subacute far contra is unreliable (r .20); far ipsi acute has no reliable map (r &minus;.03). "
     "The only significant acute &minus; pre bins are near middle&rsquo;s (decrease). Acute amplitude / own pre: near ipsi "
     "1.09, near middle 4.54, near contra .99, far ipsi 1.16, far middle .88, far contra 4.56 (inflated by the sign flip)."),
    ("fig14-2", "haufe_cue", "14", "cue",
     "Haufe decoder maps, cue window, pooled: far contra and far middle flip sign acutely across both hemispheres and "
     "return by chronic",
     "Pre-stroke each position has a reliable map (r .92&ndash;.98). Acutely far contra and far middle invert: their "
     "trials are now LESS active than the average trial over most of dorsal cortex (significant acute &minus; pre "
     "decrease over large bilateral areas), while the near positions gain (significant increases at near ipsi, middle "
     "and contra). Because the reference is the mean of all six, these are one change seen from both sides. By chronic "
     "far contra and far middle are back to their pre pattern; the residual chronic difference is a broad, untested "
     "increase at near ipsi / near contra. Acute amplitude / own pre: near ipsi 1.28, near middle 3.52, near contra "
     "1.09, far ipsi .90, far middle 1.13, far contra 2.19 (sign flip counted as amplitude)."),
    ("fig14-3", "haufe_lick", "14", "lick",
     "Haufe decoder maps, lick window, pooled: acute far contra refused (no session reached 20 lick trials); the other "
     "maps are stable",
     "Acute far contra is REFUSED: no session reached 20 far-contra lick trials, which is the deficit itself. Acute far "
     "middle rests on 3 animals / 11 sessions. Every other map keeps its pre-stroke shape after the stroke (r "
     ".75&ndash;.99); the only significant acute &minus; pre bins are two small near-middle spots. Far contra subacute "
     "and chronic look like pre. Read with Fig 14.6."),
    ("fig14-4", "haufez_precue", "z", "precue",
     "Haufe decoder maps tested against zero, pre-cue window: few significant bins in any epoch",
     "At the animal level the pre-cue maps barely clear zero. Pre: near ipsi 90 bins, near contra 52, far contra 103, "
     "the other three 0. Acute: near middle 318, near contra 123, far middle 86, far contra 0. Chronic: near ipsi 142, "
     "others &le; 42. The pre-cue decode (section 1) therefore does not localise to a consistent cortical pattern across "
     "animals with this test."),
    ("fig14-5", "haufez_cue", "z", "cue",
     "Haufe decoder maps tested against zero, cue window: far middle&rsquo;s map vanishes acutely (1,583 &rarr; 13 bins) "
     "and returns only partly",
     "Pre: every position has a significant map (near ipsi 1,672 bins, near middle 186, near contra 986, far ipsi 275, "
     "far middle 1,583, far contra 1,018). Acute: far middle 13 bins; far contra 809 bins but of the opposite sign "
     "(bilaterally blue). Subacute / chronic: far middle 52 / 105, far contra 116 / 249; near positions keep "
     "196&ndash;765. Far ipsi&rsquo;s map stays significant in every epoch (230&ndash;327)."),
    ("fig14-6", "haufez_lick", "z", "lick",
     "Haufe decoder maps tested against zero, lick window: maps persist in every epoch; far middle shrinks (1,519 &rarr; "
     "52 bins acute)",
     "Pre: near ipsi 1,716 bins, near middle 194, near contra 803, far ipsi 268, far middle 1,519, far contra 972. "
     "Post-stroke every tested panel keeps a significant map with the same sign and location; far middle shrinks to 52 "
     "bins acute (3 animals) and 70 chronic, far contra is 400 subacute and 472 chronic (acute: no data, Fig 14.3)."),
    ("fig14-7", "haufepa_precue", "pa", "precue",
     "Far-contra Haufe map per animal, pre-cue window: the acute change does not replicate (2 / 4 up, PS94 down)",
     "PS92 and PS93 turn broadly positive acutely, PS94 turns negative centrally, PS95 is mixed; PS92&rsquo;s pre map "
     "is unreliable (r .38) and PS94 subacute has none (r &minus;.13). Not a replicated effect."),
    ("fig14-8", "haufepa_cue", "pa", "cue",
     "Far-contra Haufe map per animal, cue window: the acute sign flip replicates in 4 / 4",
     "All four animals&rsquo; acute far-contra maps turn negative over bilateral cortex (PS95 from 1 acute session). "
     "Chronic: PS92, PS94 and PS95 return toward their pre map; PS93 stays negative."),
    ("fig14-9", "haufepa_lick", "pa", "lick",
     "Far-contra Haufe map per animal, lick window: the pre-stroke map replicates in 4 / 4 and is back by chronic",
     "No animal has an acute map (no session reached 20 far-contra lick trials). The pre-stroke far-contra map has the "
     "same layout in all four animals (positive left lateral, negative right anterolateral, image left = ipsi), and the "
     "subacute and chronic maps keep it; PS93&rsquo;s chronic difference is the largest."),
]
TITLE_PREFIX = {"14": "pooled", "z": "tested against zero", "pa": "far contra per animal"}


def caption(fid, png, kind, w, title, reading):
    n = fid[3:].replace("-", ".")
    head = {"14": f"Haufe decoder maps, {WTXT[w]} window, pooled, with acute &minus; pre test.",
            "z": f"Haufe decoder maps, {WTXT[w]} window, each epoch tested against zero (Musall et al. 2023 fig. S6).",
            "pa": f"Far-contra Haufe decoder map per animal, {WTXT[w]} window."}[kind]
    if kind == "14":
        rest = POOL + " " + STATS14
    elif kind == "z":
        rest = ("<b>Pooling:</b> as Fig 14.1 (per-animal pre-RMS normalisation, one colour scale). Panel titles give "
                "animals, sessions, trials and significant bins. " + STATSZ)
    else:
        rest = ("<b>Per animal:</b> far contra only; each animal&rsquo;s raw session-mean covariance (not normalised), "
                "colour scale per animal, so rows are not comparable in amplitude; difference columns = epoch &minus; pre "
                "for that animal. Panel titles give sessions, trials and split-half r. <b>Stats:</b> none (replication "
                "across animals is the evidence).")
    cap = (f'<b class="h">Fig {n} &middot; {head}</b> {DEC} <b>Window:</b> {WIN[w]}. {rest} {EPO} '
           f"<b>Reading:</b> {reading} {SCRIPT}")
    return (f'<figure id="{fid}"><div class="ftitle"><span class="fnum">Fig {n}</span>{title}</div>'
            f'<img src="{png}.png" alt="Haufe decoder maps, {TITLE_PREFIX[kind]}, {WTXT[w]} window" loading="lazy">'
            f"<figcaption>{cap}</figcaption></figure>")


sec = (
    '<section id="haufe" data-toc="14. Haufe decoder maps" data-kind="n">\n'
    '<span class="tag">Where the position code is in cortex &middot; decoder weights as activation patterns</span>\n'
    '<h2>14. Haufe decoder maps: far contra and far middle flip sign acutely after the cue, and return '
    '<span class="kind n">neural</span><span class="st prov">coupled reference</span></h2>\n'
    '<p class="input">A decoder weight is a filter, not a map: it can load heavily on a pixel that carries no signal, '
    "only to cancel shared noise (Haufe et al. 2014). The Haufe transform A = Cov(X)&nbsp;b turns each position&rsquo;s "
    "weights into the activity pattern the decoder actually reads. Decoders are refit in every session on the SVT "
    "(full-resolution pixels, no component intermediary), DAQ licks only, no kinematics. The maps are one-vs-rest, so "
    "the six positions are coupled: read a far-contra drop and a near-position gain as one event. The reference-free "
    "response maps of section 7 are the deficit measure; these say where each position&rsquo;s signal sits. "
    "<b>Not the same maps as the encoding page&rsquo;s Haufe figures</b> "
    '(<a href="https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d#f-iti_decoder_patterns_lat" target="_blank">Figs 3.69&ndash;3.71</a>): '
    "those decode continuous lick POSITION (lateral / sagittal, 3-D tongue) with a ridge, lick-locked (0&ndash;0.2 s), "
    "on each animal&rsquo;s pre-stroke 200-component basis, refit per epoch; 3.71 is one-vs-rest per spout on cued "
    "licks, without the mean-reference coupling of these maps.</p>\n"
    '<div class="take">After the cue, far contra and far middle lose their pre-stroke maps acutely and flip sign '
    "&mdash; their trials become less active than the average trial over both hemispheres "
    '(<a href="#fig14-2">Fig 14.2</a>; far contra in all four animals, <a href="#fig14-8">Fig 14.8</a>); far middle&rsquo;s map no longer clears zero acutely '
    '(<a href="#fig14-5">Fig 14.5</a>). Both return toward their pre-stroke pattern by chronic. In the lick window the '
    "maps keep their pre-stroke layout whenever there are enough far-contra licks to fit one "
    '(<a href="#fig14-3">Fig 14.3</a>, <a href="#fig14-9">14.9</a>). The pre-cue maps are too weak to localise '
    '(<a href="#fig14-4">Fig 14.4</a>) and their acute change does not replicate (<a href="#fig14-7">Fig 14.7</a>).</div>\n'
    + "\n".join(caption(*f) for f in FIGS) + "\n</section>\n\n"
)
anchor = '<section id="closed"'
assert s.count(anchor) == 1
s = s.replace(anchor, sec + anchor)

# No 'Where things stand' row (Priya 10-27): these maps are a coupled reference, not a deficit measure, and the
# cue-window acute loss they show is already carried by section 7. The section is linked from its own take-home.
P.write_text(s, encoding="utf8")
print("section 14 added:", len(FIGS), "figures")

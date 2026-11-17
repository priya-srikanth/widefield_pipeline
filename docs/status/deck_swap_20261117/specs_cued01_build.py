"""Build specs/cued01.json: cued per-spout response AUC moved to iti_licks.CUED_RESP_WIN (0..1.0 s; Priya 11-17 "yes update
113 and 114 to match"; DECISIONS 2026-11-17). "old" strings come from a FRESH scratch dump of the deck (13:35 version) and
are asserted. Numbers: cued_ramp_cued01_20261117 render (perspout_auc_tests.csv, identical cell values to the rebuild's
l_cued_ramp_g table, which already held the 0-1 window) and the lmm_hidden cued01 render."""
import json
import sys
from pathlib import Path

D = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
HERE = Path(__file__).parent
SW = "M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-17/swap_inputs/"
TAG = "hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk"
IL = "M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/"
LMM0 = f"M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/grant_figures/{TAG}/lmm_hidden_20261117/"
LMM = LMM0 + "cued01_20261117/"
LABELS = ("Window", "Licks", "Trial period", "Trial types", "Gate", "Baseline", "Correction")


def title(n):
    return next(p for sh in D[str(n)]["shapes"] for p in sh.get("paras", []) if p.strip())


def para(n, shape, start):
    for sh in D[str(n)]["shapes"]:
        if sh["name"] == shape:
            for p in sh.get("paras", []):
                if p.startswith(start):
                    return p
    raise SystemExit(f"slide {n}: no paragraph in {shape} starting {start!r}")


def data_and_body(n):
    notes = [p for p in D[str(n)]["notes"] if p.strip()]
    data = {}
    for p in notes[:7]:
        k, v = p.split(": ", 1)
        assert k in LABELS, (n, k)
        data[k] = v
    return data, notes[7:]


def sub(text, reps, n):
    for a, b in reps:
        assert a in text, (n, a[:80])
        text = text.replace(a, b)
    return text


specs = []
# ------------------------------------------------------------------------------------------------ 113 (Fig 14.7, KEY)
d, body = data_and_body(113)
d["Window"] = ("first cued lick -2..+3 s (cue-aligned version -2..+3 s); response AUC 0..1.0 s (cued response window, "
               "iti_licks.CUED_RESP_WIN; Priya 11-17), baseline -1.0..-0.5 s")
d["Correction"] = ("AUC: paired nested bootstrap 95 % CI (boot_delta), uncorrected; curve bars: per-bin bootstrap, BH-FDR q .05 "
                   "over lags [traced: render log cued_ramp_cued01_20261117/logs/cued_ramp.log]")
old0 = ("Response AUC 0-0.5 s / the animal's pre AUC at the same spout, change vs pre, acute / subacute / chronic, "
        "contralesional tjM1 (* = CI excludes 0): close ipsi +.57* / +.47* / +.35*, close mid +.61* / +.50* / +.37*, close "
        "contra +.75* / +.47* / +.37*, far ipsi +.56* / +.49* / +.63*, far mid +.27* / +.52* / +.64*, far contra +.20 / "
        "+.29 / +.52* (4 animals each). The cue-aligned version: far contra acutely down in cortex (-.52*, 0-0.5 s) and "
        "SSp-m contra (0-1 s -.53*), n.s. in ALM / tjM1 contra.")
new0 = ("Response AUC 0-1.0 s / the animal's pre AUC at the same spout, change vs pre, acute / subacute / chronic, "
        "contralesional tjM1 (* = CI excludes 0): close ipsi +.72* / +.46* / +.37*, close mid +.74* / +.47* / +.34*, close "
        "contra +.91* / +.43* / +.33*, far ipsi +.47* / +.36* / +.55*, far mid +.13 / +.49* / +.64*, far contra -.12 / "
        "+.22 / +.48* (4 animals each). The cue-aligned version: far contra acutely down in cortex (-.52*) and SSp-m contra "
        "(-.53*), n.s. in ALM / tjM1 contra.")
body = [sub(body[0], [(old0, new0),
                      ("Mean +- SEM over sessions with >= 5 licks of that spout;",
                       "Curves / the animal's pre response AUC 0..1.0 s over all spouts; mean +- SEM over sessions with >= 5 "
                       "licks of that spout;")], 113), *body[1:]]
body = [p if not p.startswith("FIGURE SOURCE") else
        (f"FIGURE SOURCE: Encoding (goal) page Fig 14.7 [#f-lick_cued_ramp_perspout] (https://claude.ai/artifact/"
         f"Fnpn2KHknMwh3r13xADH7d) · render {IL}cued_ramp_cued01_20261117/{TAG}/figures/lick_cued_ramp_perspout.png "
         "(cued per-spout window 0..1.0 s, re-render 11-17 on the rebuild cached traces) · deck image: presentations/"
         "_source_PI_overview_2026-11-17/swap_inputs/s113_cued_perspout_cued01.png (caption removed)") for p in body]
specs.append({
    "slide": 113, "title_now": title(113), "status": "ready", "skip_reason": "",
    "pictures": [{"shape_name": "fig:lick_cued_ramp_perspout", "new_image": SW + "s113_cued_perspout_cued01.png",
                  "source_render": f"{IL}cued_ramp_cued01_20261117/{TAG}/figures/lick_cued_ramp_perspout.png",
                  "crop_done": "caption (suptitle) cut (top 138 px) + purple footer whited out, as the old image; 2200 x 962"}],
    "slide_text": [{"shape_name": "TextBox 4", "old": para(113, "TextBox 4", "First cued lick by target spout"),
                    "new": "First cued lick by target spout, contralesional areas; / the animal's pre response AUC 0-1 s "
                           "over all spouts; bars = bins that differ from pre at that spout"}],
    "notes_body": body, "data_block": d,
    "changes": [{"what": "tjM1 contra per-spout AUC / pre at spout, acute", "old": "close +.57* / +.61* / +.75*, far ipsi +.56*, "
                 "far mid +.27*, far contra +.20 (0..0.5 s)", "new": "close +.72* / +.74* / +.91*, far ipsi +.47*, far mid +.13 "
                 "(n.s.), far contra -.12 (n.s.) (0..1.0 s)", "source": "cued_ramp_cued01_20261117/perspout_auc_tests.csv"},
                {"what": "cued per-spout cells with * (4 ROIs x 6 spouts x 3 epochs)", "old": "60 of 72", "new": "49 of 72",
                 "source": "same"},
                {"what": "curve scale", "old": "/ pre response AUC 0..0.5 s", "new": "/ pre response AUC 0..1.0 s (per-bin "
                 "bars: 2187 -> 2178 significant lick bins, 17 bins differ)", "source": "sig_bins_perspout.csv"}],
    "conclusion_changed": False,
    "conclusion_note": "Title holds and sharpens: acutely the far-contra response is now below pre (n.s.) and far mid loses "
                       "its star; close spouts are larger. Chronic far contra stays up (+.48*)."})

# ------------------------------------------------------------------------------------------------ 114 (LMM companion)
d, body = data_and_body(114)
d["Window"] = ("lick-aligned, per-trial baseline -1.5..-1.0 s, response baseline -1.0..-0.5 s; response AUC 0-1.0 s per "
               "spout (cued response window, Priya 11-17) / the animal's pre AUC at that spout and ROI")
d["Correction"] = d["Correction"].replace("lmm_hidden_20261117/logs/cued_ramp_perspout.log",
                                          "lmm_hidden_20261117/cued01_20261117/logs/cued_ramp_perspout.log")
assert "cued01_20261117/logs/cued_ramp_perspout.log" in d["Correction"]
starred = ("subacute ALM contra close ipsi *, subacute ALM contra close mid *, subacute ALM contra close contra *, subacute "
           "ALM contra far mid *, subacute tjM1 contra close ipsi *, chronic tjM1 contra close ipsi *, subacute tjM1 contra "
           "close mid *, chronic tjM1 contra close mid *, subacute tjM1 contra close contra *, chronic tjM1 contra close "
           "contra *, subacute tjM1 contra far mid *, chronic tjM1 contra far contra *, subacute SSp-m contra close ipsi *, "
           "chronic SSp-m contra close ipsi *, subacute SSp-m contra close mid *, chronic SSp-m contra close mid *, subacute "
           "SSp-m contra close contra *, chronic SSp-m contra close contra *, subacute SSp-m contra far ipsi *, subacute "
           "SSp-m contra far mid *, chronic SSp-m contra far contra *, subacute cortex close ipsi *")
nb = []
for p in body:
    p = p.replace("First cued lick response per spout, AUC 0-0.5 s / pre, 4 ROIs x 6 spouts.",
                  "First cued lick response per spout, AUC 0-1.0 s / pre (cued window, Priya 11-17), 4 ROIs x 6 spouts.")
    if p.startswith("Starred cells:"):
        p = f"Starred cells: {starred}."
    if p.startswith("FIGURE SOURCE"):
        p = (f"FIGURE SOURCE: {SW}s114_lick_cued_ramp_perspout_cued01_lmm.png (title cut of {LMM}lick_cued_ramp_perspout_"
             f"cued01.png); statistics {LMM}lick_cued_ramp_perspout_cued01_lmm.csv; inputs {LMM}lick_cued_ramp_perspout_"
             f"cued01_sessions.csv; source table {IL}{TAG}/cued_wide/*.npz + sessions_roi8fr; reproduces {IL}{TAG}/"
             "perspout_auc_tests.csv (window 0-1) 72/72 cells. Code: wf_lmm_main branch lmm-hidden-20261103 "
             f"scripts/lmm_hidden/ (render log {LMM}logs/cued_ramp_perspout.log carries the code sha and env).")
    nb.append(p)
assert any("AUC 0-1.0 s / pre (cued window" in p for p in nb)
specs.append({
    "slide": 114, "title_now": title(114), "status": "ready", "skip_reason": "",
    "pictures": [{"shape_name": "fig:lick_cued_ramp_perspout_lmm", "new_image": SW + "s114_lick_cued_ramp_perspout_cued01_lmm.png",
                  "source_render": LMM + "lick_cued_ramp_perspout_cued01.png",
                  "crop_done": "title cut (crop.drop_title), as the old notitle image; same pixel size 3599 x 1190"}],
    "slide_text": [{"shape_name": "TextBox 3", "old": para(114, "TextBox 3", "•  31 of 72"),
                    "new": "•  22 of 72 tested cells starred (0 **)"},
                   {"shape_name": "TextBox 3", "old": para(114, "TextBox 3", "•  acute SSp-m contra close ipsi *"),
                    "new": "•  no acute cell; subacute ALM contra close ipsi / mid / contra / far mid *, subacute + chronic "
                           "tjM1 and SSp-m contra close spouts *, chronic tjM1 / SSp-m contra far contra * (+ notes)"}],
    "notes_body": nb, "data_block": d,
    "changes": [{"what": "LMM starred cells (72 tests)", "old": "31 (0..0.5 s)", "new": "22 (0..1.0 s); dropped: acute tjM1 far "
                 "ipsi, acute SSp-m close ipsi, acute cortex far ipsi, subacute tjM1 far ipsi, cortex close spouts subacute / "
                 "chronic (5 cells); none added", "source": LMM + "logs/cued_ramp_perspout.log"}],
    "conclusion_changed": True,
    "conclusion_note": "Hidden backup: the mixed model no longer stars any ACUTE per-spout cell (3 at 0..0.5 s); the subacute "
                       "/ chronic close-spout increases hold."})

out = HERE / "specs" / "cued01.json"
if out.exists():
    raise SystemExit(f"refusing to overwrite {out}")
out.write_text(json.dumps(specs, indent=1, ensure_ascii=False), encoding="utf-8")
print(out, len(specs), "slides")

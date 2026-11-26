"""Page section (CC-B follow-up 11-25): six-spout decomposition + the frozen-frame Figs 17.1 / 17.2 proper."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

AN = ["PS92", "PS93", "PS94", "PS95"]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
PL = {"close_ipsi": "close ipsi", "close_mid": "close mid", "close_contra": "close contra", "far_ipsi": "far ipsi",
      "far_mid": "far mid", "far_contra": "far contra"}
REG = ["SSp-m_left", "SSp-m_right", "cortex"]
RL = {"SSp-m_left": "SSp-m ipsi", "SSp-m_right": "SSp-m contra", "cortex": "cortex"}


def _fmt(x):
    return f"{x:+.1f}".replace("-", "&minus;")


def pooled_grid(D: Path, run: str, part: str, epoch: str) -> str:
    P = pd.read_csv(D / run / "pooled.csv")
    q = P[(P.part == part) & (P.epoch == epoch)]
    rows = []
    for reg in REG:
        cells = []
        for pos in POS:
            r = q[(q.region == reg) & (q.pos == pos)].iloc[0]
            star = "<b class=star>*</b>" if (r.lo > 0 or r.hi < 0) else ""
            cells.append(f"<td>{_fmt(r.point)}{star} [{_fmt(r.lo)}, {_fmt(r.hi)}] ({int(r.n_same_sign)}/{int(r.n_animals)})</td>")
        rows.append(f"<tr><td class=t>{RL[reg]}</td>{''.join(cells)}</tr>")
    return ("<table class=num><thead><tr><th></th>" + "".join(f"<th>{PL[p]}</th>" for p in POS) + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def per_animal_grid(D: Path, run: str, part: str, epoch: str, reg: str) -> str:
    R = pd.read_csv(D / run / "per_animal.csv")
    q = R[(R.part == part) & (R.epoch == epoch) & (R.region == reg)]
    rows = []
    for an in AN:
        cells = "".join(f"<td>{_fmt(q[q.pos == pos][an].iloc[0])}</td>" for pos in POS)
        rows.append(f"<tr><td class=t>{an}</td>{cells}</tr>")
    return ("<table class=num><thead><tr><th></th>" + "".join(f"<th>{PL[p]}</th>" for p in POS) + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


R_PS_LICK = ("Per-session fits, lick-aligned: the acute response RISES at every close spout and at far ipsi (SSp-m +11 to +24 x1e3, "
             "4/4), is flat at far mid and FALLS only at far contra (ipsi SSp-m &minus;20.9*, 4/4; contra &minus;7.0, 3/4); the lick part "
             "tracks the measured change at every spout (close spouts +13 to +24*, far contra &minus;23.9* / &minus;10.4*, 4/4) and the "
             "target part is ~0 at all six (|change| &le; 2.4 x1e3, no 4/4 cell): the far-contra fall is spout-specific, and it is the "
             "lick part at that spout, not a loss of target coding anywhere.")
R_PS_CUE = ("Cue-aligned: the same pattern with larger numbers (far contra measured &minus;29.8* / &minus;17.2*, lick part &minus;30.2* / "
            "&minus;19.5*, 4/4; close spouts +13 to +28*); the target part has two small negative far-contra cells (&minus;2.9* ipsi, "
            "&minus;2.3* contra) against a lick part ten times larger.")
R_FR_LICK = ("Frozen frame A, lick-aligned: the frozen lick part FALLS acutely at every spout, graded by distance (close &minus;1.4 to "
             "&minus;2.7*, far ipsi &minus;2*, far mid &minus;5*, far contra &minus;10.6* / &minus;9.9*, 4/4), the frozen target part is ~0 "
             "everywhere (|change| &le; 0.6 x1e3), and the frozen residual is large and POSITIVE at all six spouts (+7 to +29*, 4/4): the "
             "frozen frame predicts less lick-driven activity everywhere while the measured activity rose at the near spouts, so the "
             "far-contra specificity is in the measured response (and in the per-session lick part), not in the frozen lick part.")
R_FR_CUE = ("Cue-aligned frozen frame A: lick part &minus;9.1* / &minus;8.7* at far contra (4/4), &minus;4.8* far mid, &minus;3.2* close "
            "contra, about &minus;1 at the other close spouts; target part within &plusmn;0.2 x1e3 at every spout; residual positive at every "
            "spout except far contra (+1.8 / +4.5, n.s.).")
R_171 = ("Fig 17.1 proper on wave-1 frame A (rebuild_figs/fig_pooled.py 4p block, unchanged): acute far-contra SSp-m measured &minus;0.49 "
         "ipsi / &minus;0.25 contra of pre, predicted &minus;0.67 / &minus;0.64 (4/4), of which lick &minus;0.24 / &minus;0.33 (4/4) and target "
         "+0.01 / &minus;0.02; residual +0.17 / +0.39 (4/4); chronic measured +0.26 / +0.45 with predicted ~0 and residual +0.37 / +0.42 "
         "(4/4). v36 frame with the same reader: lick &minus;0.17 / &minus;0.26, target +0.03 / &minus;0.00, residual +0.14 / +0.36 acute; "
         "+0.32 / +0.39 chronic.")
R_172 = ("Bars = change from pre per frozen part (dF/F x1e3), lick window; drawn by scripts/g4figs_1125/g4_decomp_fig.py --frozen (no "
         "repo script draws the page's fig4_decomp_epochs; it was a 10-24 scratchpad figure of the same table).")
ANSWER = ("<div class=take><p><b>Answer.</b> Frozen frame and per-session fits agree on the two claims: the target part is unchanged "
          "acutely at far contra in every animal (per-session +0.01 / &minus;0.05 of pre ipsi / contra, frozen +0.01 / &minus;0.02; per "
          "animal all within &plusmn;0.04 except PS94 contra &minus;0.14 per-session / &minus;0.04 frozen), and the lick part carries the "
          "acute far-contra drop in every animal (per-session &minus;0.56 / &minus;0.35 of pre, 4/4; frozen &minus;0.24 / &minus;0.33, 4/4; "
          "PS92 &minus;0.81 / &minus;0.72 vs &minus;0.28 / &minus;0.42, PS93 &minus;0.44 / &minus;0.36 vs &minus;0.19 / &minus;0.22, PS94 "
          "&minus;0.53 / &minus;0.07 vs &minus;0.22 / &minus;0.28, PS95 &minus;0.48 / &minus;0.25 vs &minus;0.26 / &minus;0.40). They differ "
          "in what else the frozen frame predicts: its lick part falls at all six spouts (graded, strongest at far contra) and its full "
          "prediction overshoots the measured far-contra drop (&minus;0.64 vs &minus;0.25 contra), leaving a positive residual at every "
          "spout. Across spouts the acute fall is specific to far contra in the MEASURED response and in the per-session lick part (every "
          "close spout and far ipsi rise, 4/4); it is not a far-contra-specific loss in the frozen lick part.</p></div>")


def section(page: Path, fig_block, details, prov, lineage: dict) -> str:
    D = page / "tables" / "decomp_allspouts"
    figs = page / "figs"
    out = ["<section id=allspouts><h2>Follow-up: all six spouts, and the frozen-frame Figs 17.1 / 17.2 proper</h2>",
           "<p class=sub>Question (11-25 evening): is the acute lick-part fall specific to far contra, or present at every spout? "
           "Per-session G4 fits and the FROZEN pre-stroke frame A (wave 1, four-camera root) decomposed with the same reader "
           "(scripts/lp_encoder_decompose.py, load-only) at all six spouts, lick- and cue-aligned; per animal first, then pooled "
           "(nested bootstrap). Values = evoked dF/F x1e3, epoch minus the animal's own August pre; * = 95% CI excludes 0; "
           "(k/4) = animals with the sign of the pooled mean.</p>"]
    common = {"Reader": "scripts/lp_encoder_decompose.py (--align lick: 0..+1 s after the first post-cue lick minus &minus;0.3..&minus;0.1 s; "
                        "--align cue: +0.5..+1.5 s minus &minus;0.5..0 s) -> scripts/g4figs_1125/g4_decomp_allspouts.py (per animal, boot_delta seed 0)",
              "Per-session parts": "measured = data; lick part = instructed group (cued licks, executed angle, 3-D tongue, jaw); target part = position-specific part of task; uninstructed = nose, running, spontaneous licks; residual = data &minus; full",
              "Frozen parts": "measured; lick part = frozen lick kernels; target part = frozen target block; other = spontaneous-lick / other; residual = data &minus; full prediction of the frozen pre-stroke frame (saved parts overlap: fr, contact, lick_all, spout are groupings of the same columns)",
              **{k: lineage[k] for k in ("Lineage root", "Licks", "Pose / 3-D", "Trials", "Baseline")}}
    for al, lab, reading in (("lick", "F1", R_PS_LICK), ("cue", "F2", R_PS_CUE)):
        run = f"persession_G4_{al}"
        img = f"figs/decomp_allspouts_persession_G4_{al}.png"
        shutil.copy2(D / run / "decomp_allspouts.png", figs / Path(img).name)
        lp = "lick part (instructed)"
        body = "<h4>Acute minus pre, pooled, lick part</h4>" + pooled_grid(D, run, lp, "acute")
        body += "<h4>Acute, pooled, measured</h4>" + pooled_grid(D, run, "measured", "acute")
        body += "<h4>Acute, pooled, target part</h4>" + pooled_grid(D, run, "target part", "acute")
        body += "<h4>Acute, pooled, residual</h4>" + pooled_grid(D, run, "residual", "acute")
        for reg in REG:
            body += f"<h4>Acute lick part per animal, {RL[reg]}</h4>" + per_animal_grid(D, run, lp, "acute", reg)
        body += "<h4>Chronic minus pre, pooled, lick part</h4>" + pooled_grid(D, run, lp, "chronic")
        body += "<h4>Chronic, pooled, residual</h4>" + pooled_grid(D, run, "residual", "chronic")
        out.append(fig_block(lab, f"Six-spout decomposition, per-session G4 fits, {al}-aligned", reading, img,
                             {**common, "Tag": "main (per-session, thr_push)",
                              "File": f"tables/decomp_allspouts/{run}/per_animal.csv, pooled.csv; v36 twin in persession_v36_{al}"},
                             klass="b", old_vs_new=body))
    for al, lab, reading in (("lick", "F3", R_FR_LICK), ("cue", "F4", R_FR_CUE)):
        run = f"frozen_frA_{al}"
        img = f"figs/decomp_allspouts_frozen_frA_{al}.png"
        shutil.copy2(D / run / "decomp_allspouts.png", figs / Path(img).name)
        body = "<h4>Acute minus pre, pooled, lick part (frozen)</h4>" + pooled_grid(D, run, "lick part", "acute")
        body += "<h4>Acute, pooled, measured</h4>" + pooled_grid(D, run, "measured", "acute")
        body += "<h4>Acute, pooled, target part (frozen)</h4>" + pooled_grid(D, run, "target part", "acute")
        body += "<h4>Acute, pooled, residual (frozen)</h4>" + pooled_grid(D, run, "residual", "acute")
        for reg in REG:
            body += f"<h4>Acute lick part per animal, {RL[reg]}</h4>" + per_animal_grid(D, run, "lick part", "acute", reg)
        body += "<h4>v36 frame, acute lick part, pooled (same reader)</h4>" + pooled_grid(D, f"frozen_v36_{al}", "lick part", "acute")
        body += "<h4>Chronic minus pre, pooled, residual (frozen)</h4>" + pooled_grid(D, run, "residual", "chronic")
        out.append(fig_block(lab, f"Six-spout decomposition, FROZEN frame A (wave 1, four-camera), {al}-aligned", reading, img,
                             {**common, "Frame": "rebuild_20261123_fourcam/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding/&lt;run tag&gt;/frozen_encoder_gate-attempted_v7g3_t3d_spont_frA/joint/sessions (load-only; NB the session json lick_set field reads cam4 while LINEAGE.txt says fused four-camera licks: a field label to check)",
                              "File": f"tables/decomp_allspouts/{run}/per_animal.csv, pooled.csv; v36 frame twin frozen_v36_{al}"},
                             klass="b", old_vs_new=body))
    for n, src, title, reading in (
        ("17.1 proper", page / "tables" / "frozen_17_frA" / "fig4p_decomp_pooled.png", "Encoder parts: pooled change, FROZEN frame A (the page's reader)", R_171),
        ("17.2 proper", page / "tables" / "frozen_17_frA" / "fig17_2_decomp_epochs_persession.png", "SSp-m far-contra change by part, FROZEN frame A", R_172),
    ):
        img = f"figs/{Path(src).stem}_frA.png"
        shutil.copy2(src, figs / Path(img).name)
        out.append(fig_block(n, title, reading, img,
                             {**common, "Frame": "wave-1 frame A, four-camera root (see F3); v36 twin in tables/frozen_17_v36/",
                              "Script": "scripts/lp_encoder_decompose.py --folder frozen_encoder_gate-attempted_v7g3_t3d_spont_frA/joint --align lick -> rebuild_figs/fig_pooled.py (4p) via g4_pooled_4p.py; g4_decomp_fig.py --frozen",
                              "File": "tables/frozen_17_frA/pooled_numbers_4p.txt, fig17_1_table.csv, fig17_2_table.csv"}, klass="b"))
    out.append(ANSWER + "</section>")
    return "".join(out)

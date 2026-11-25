"""Build the ONE local HTML page of the G4-fed Movement (Execution) Coding figures (CC-B helper, 11-25).

Reads the rendered figures + tables (G4 = per-session combined_encoding fits enc_fourcam3d_20261124, and the same
readers run on the v36 collected fits for old-vs-new), copies the figures next to the page and writes
index_g4_key_results_20261125.html. Local file only (never published). American English; PS92-95 IDs.
"""
from __future__ import annotations

import html
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PAGE = Path(r"M:/MICROSCOPE/Priya/Widefield/run_records/2026-11-25/g4_key_results_ccB1125")
OUT = Path(r"M:/MICROSCOPE/Priya/Widefield/rebuild_20261123_fourcam/figures_working/g4figs_ccB_20261125")
TAG = ("hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-"
       "stopA2jexrt3500c70_daqlk")
G4 = OUT / "imaging_encoding_view" / TAG
V36 = Path(r"M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/"
           r"imaging_encoding") / TAG
V36V = OUT / "imaging_encoding_view_v36" / TAG
BUNDLE = "M:/MICROSCOPE/Priya/Widefield/rebuild_20261123_fourcam/o2/enc_fourcam3d_20261124"
CMP = Path(r"M:/MICROSCOPE/Priya/Widefield/run_records/2026-11-24/g4_running_compare_ccB1124")
FIGS = PAGE / "figs"
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
POST = EP[1:]

LINEAGE = {
    "Lineage root": "M:/MICROSCOPE/Priya/Widefield/rebuild_20261123_fourcam (four-camera 3-D root); LP run "
                    "lp_r4_heatmap_20261013__clean_v37_sync60; run tag " + TAG,
    "Encoder fits": f"O2 bundle {BUNDLE} (enc_fourcam3d_20261124, code 05bfc260; 20 tags x 107 sessions, collected on "
                    "the share under out/&lt;tag&gt;/sessions; read in place, not copied into the routed encoding folder)",
    "Licks": "fused four-camera lick list lp_side_r2b_prod_20261118_3net_20261121_sync60_gapfix60 (COMBINED_ENCODING_LICKS="
             "fourcam, LICKS_ALL=split: cued licks and spontaneous pre-cue / ITI licks in separate kernels)",
    "Pose / 3-D": "session_poses/pose3d_4cam_3net_20261121_sync60: tongue, jaw and nose 3-D (xyz3d layout), executed "
                  "angle polar from the 3-D tongue; spout targets from pose3d_spout_positions_commanded_20261104.csv",
    "Trials": "jaw rule 3d_mouth (jaw_onsets_cap_ccB1123.csv, s1_trial_classes_cue_stop.csv.gz); camera working period "
              "(WIDEFIELD_WORKING_TRIALS=in_working: contact + incomplete, jaw-only out), trials cue to stop, stop = last "
              "lick; attempted gate (OROFACIAL_GATE=attempted, trial_attempts.csv sha 99bc7c1b)",
    "Baseline": "August-only pre (WIDEFIELD_PRE_FROM=0801; June pre sessions dropped by every reader); each readout = "
                "the animal's own pre",
    "Running model": "thr_push (primary, Priya 11-24); main_spline / main_thr_pushhinge = sensitivity rows (not drawn)",
    "Hemo / basis": "meegkit_hpfit_lag_refitT_swapmask, LocaNMF projection, area basis splitr8",
}


def esc(s) -> str:
    return html.escape(str(s), quote=False)


def fmt(x, nd=3, sign=False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "&mdash;"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "&minus;")


def table(df: pd.DataFrame, nd=3, sign=False, index=True, cls="num") -> str:
    df = df.copy()
    cols = list(df.columns)
    h = "".join(f"<th>{esc(c)}</th>" for c in ([""] if index else []) + cols)
    rows = []
    for ix, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f"<td>{fmt(float(v), nd, sign)}</td>")
            elif isinstance(v, (int, np.integer)):
                cells.append(f"<td>{int(v)}</td>")
            else:
                cells.append(f"<td class=t>{esc(v)}</td>")
        rows.append("<tr>" + (f"<td class=t>{esc(ix)}</td>" if index else "") + "".join(cells) + "</tr>")
    return f"<table class={cls}><thead><tr>{h}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def ci(point, lo, hi, nd=3, star=True) -> str:
    s = f"{fmt(point, nd, True)} [{fmt(lo, nd, True)}, {fmt(hi, nd, True)}]"
    if star and np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0):
        s += "<b class=star>*</b>"
    return s


def details(title: str, body: str, open_=False) -> str:
    return f"<details class=capinfo{' open' if open_ else ''}><summary>{title}</summary>{body}</details>"


def prov(rows: dict) -> str:
    return "<table class=provtab>" + "".join(f"<tr><th>{esc(k)}</th><td>{v}</td></tr>" for k, v in rows.items()) + "</table>"


def copy_fig(src: Path, name: str) -> str:
    FIGS.mkdir(parents=True, exist_ok=True)
    dst = FIGS / name
    if not src.exists():
        return ""
    shutil.copy2(src, dst)
    return f"figs/{name}"


# ----------------------------------------------------------------------------------------------- numbers
def pooled_numbers(path: Path) -> dict:
    d = {}
    for ln in path.read_text().splitlines():
        k, v = ln.split(":", 1)
        p, rest = v.strip().split(" [")
        lo, hi = rest.rstrip("]").split(", ")
        d[k.strip()] = (float(p), float(lo), float(hi))
    return d


def contest_tables() -> tuple[str, str]:
    T = pd.read_csv(PAGE / "tables" / "contest" / "contest_per_animal.csv")
    blocks = []
    for lin in ("G4", "v36"):
        for w in ("lick", "cue"):
            g = T[(T.lineage == lin) & (T.width == 1) & (T.window == w)]
            P = g.pivot_table(index="animal", columns="epoch", values="contest").reindex(columns=EP) * 1e3
            D = P[POST].sub(P["pre"], axis=0)
            D.columns = [f"{e} &minus; pre" for e in POST]
            N = g.pivot_table(index="animal", columns="epoch", values="n").reindex(columns=EP).astype(int)
            N.columns = [f"n {e}" for e in EP]
            blocks.append(f"<p class=small><b>{lin}, {w} window, width 1</b>: contest (target &minus; executed) unique "
                          f"dR&sup2; x1e3, per animal (mean of sessions) then change from own pre; n = sessions.</p>"
                          + table(pd.concat([P.round(2), D.round(2), N], axis=1), nd=2, sign=False))
    g4 = pooled_numbers(OUT / "figs" / "pooled_numbers_1p_10p.txt")
    v36 = pooled_numbers(PAGE / "v36_same_reader" / "pooled_numbers_1p_10p.txt")
    rows = []
    for k in g4:
        rows.append({"readout": k, "G4 (four-camera)": ci(*g4[k], nd=4), "v36 (old lineage, same reader)": ci(*v36[k], nd=4)})
    pooled = pd.DataFrame(rows).set_index("readout")
    pooled_html = "<table class=num><thead><tr><th>readout</th><th>G4 (four-camera)</th><th>v36 (same reader)</th></tr></thead><tbody>" + "".join(
        f"<tr><td class=t>{esc(i)}</td><td>{r.iloc[0]}</td><td>{r.iloc[1]}</td></tr>" for i, r in pooled.iterrows()) + "</tbody></table>"
    return "".join(blocks), pooled_html


def shares_tables() -> str:
    T = pd.read_csv(PAGE / "tables" / "shares" / "shares_by_animal_epoch.csv")
    P = pd.read_csv(PAGE / "tables" / "shares" / "shares_change_vs_pre.csv")
    out = []
    for w in ("lick", "cue", "all"):
        for q, name in (("uniq_target", "unique target (nokin:u:target)"), ("uniq_exec", "unique executed (nokin:u:exec)"),
                        ("drop_position", "position drop (full &minus; drop:position)"), ("full", "full model R&sup2;")):
            t = T[(T.window == w) & (T.quantity == q)].pivot_table(index="animal", columns="tag", values=EP, aggfunc="first")
            t = t.reindex(["PS92", "PS93", "PS94", "PS95", "mean"])
            t.columns = [f"{tag.replace('main_', '')} {ep}" for ep, tag in t.columns]
            t = t[[f"{tag} {ep}" for tag in ("main", "spline") for ep in EP]]
            p = P[(P.window == w) & (P.quantity == q)]
            pr = {tag: {ep: ci(*p[(p.tag == tag) & (p.epoch == ep)][["point", "lo", "hi"]].iloc[0] * 1e3, nd=2)
                        + f" ({int(p[(p.tag == tag) & (p.epoch == ep)].n_up.iloc[0])}/4 up)" for ep in POST} for tag in ("main", "main_spline")}
            pooled = "<table class=num><thead><tr><th>change vs own pre (x1e3, 95% nested CI)</th>" + "".join(f"<th>{e}</th>" for e in POST) + "</tr></thead><tbody>" + "".join(
                f"<tr><td class=t>{tag}</td>" + "".join(f"<td>{pr[tag][e]}</td>" for e in POST) + "</tr>" for tag in pr) + "</tbody></table>"
            out.append(f"<h4>{w} window: {name}</h4><p class=small>Per animal (mean of sessions) x1e3; main = thr_push "
                       f"(primary), spline = main_spline (sensitivity).</p>" + table((t * 1e3).round(2), nd=2) + pooled)
    return "".join(out)


def penalties_tables() -> str:
    S = pd.read_csv(PAGE / "tables" / "penalties" / "penalties_by_animal_epoch.csv")
    out = []
    for lin in ("G4", "v36"):
        s = S[(S.lineage == lin) & (S.tag == "main") & (S.model == "full")]
        t = s.pivot_table(index=["group", "animal"], columns="epoch", values="median_log10").reindex(columns=EP)
        q = s.pivot_table(index=["group", "animal"], columns="epoch", values="q25").reindex(columns=EP)
        q2 = s.pivot_table(index=["group", "animal"], columns="epoch", values="q75").reindex(columns=EP)
        n = s.pivot_table(index=["group", "animal"], columns="epoch", values="n").reindex(columns=EP)
        rows = []
        for ix in t.index:
            r = {"group": ix[0], "animal": ix[1]}
            for ep in EP:
                if np.isfinite(t.loc[ix, ep]):
                    r[ep] = f"{t.loc[ix, ep]:.1f} ({q.loc[ix, ep]:.0f}&ndash;{q2.loc[ix, ep]:.0f}; n {int(n.loc[ix, ep])})"
                else:
                    r[ep] = "&mdash;"
            rows.append(r)
        df = pd.DataFrame(rows).set_index(["group", "animal"])
        body = "<table class=num><thead><tr><th>group</th><th>animal</th>" + "".join(f"<th>{e}</th>" for e in EP) + "</tr></thead><tbody>" + "".join(
            f"<tr><td class=t>{g}</td><td class=t>{a}</td>" + "".join(f"<td>{df.loc[(g, a), e]}</td>" for e in EP) + "</tr>" for g, a in df.index) + "</tbody></table>"
        out.append(f"<h4>{lin}: full model, median log10(penalty) per animal x epoch (IQR; n sessions)</h4>" + body)
    return "".join(out)


def fr_pooled(path: Path) -> dict:
    d = {}
    for ln in path.read_text().splitlines():
        k, v = ln.rsplit(":", 1)
        p, rest = v.strip().split(" [")
        lo, hi = rest.rstrip("]").split(", ")
        d[k.strip()] = (float(p), float(lo), float(hi))
    return d


def fr_summary_table(areas=("cortex", "SSp-m_left", "SSp-m_right")) -> str:
    g4 = fr_pooled(G4 / "fr_encoding_summary" / "pooled_changes.txt")
    v = fr_pooled(V36 / "fr_encoding_summary" / "pooled_changes.txt")
    rows = []
    for ar in areas:
        for w in ("cue", "lick"):
            for q in ("position, no FR", "position | FR (A)", "FR gain (A)"):
                for ep in POST:
                    k = f"{ar} {w} {q} {ep}-pre"
                    if k in g4:
                        rows.append(f"<tr><td class=t>{ar}</td><td class=t>{w}</td><td class=t>{q}</td><td class=t>{ep}</td>"
                                    f"<td>{ci(*g4[k])}</td><td>{ci(*v.get(k, (np.nan,) * 3))}</td></tr>")
    return ("<table class=num><thead><tr><th>area</th><th>window</th><th>quantity</th><th>epoch</th><th>G4</th><th>v36</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def fr_per_animal(area="cortex", w="cue") -> str:
    out = []
    for lin, p in (("G4", G4), ("v36", V36)):
        d = pd.read_csv(p / "fr_encoding_summary" / "fr_summary_by_animal.csv")
        d = d[(d["mode"] == "A") & (d.area == area) & (d.window == w)]
        rows = []
        for q in ("pos", "pos_fr", "fr_gain"):
            t = d.pivot_table(index="animal", columns="epoch", values=q).reindex(columns=EP)
            dd = t[POST].sub(t.pre, axis=0) * 1e3
            dd.insert(0, "pre level x1e3", t.pre * 1e3)
            dd.index = [f"{q} {a}" for a in dd.index]
            rows.append(dd)
        out.append(f"<p class=small><b>{lin}</b>, {area}, {w} window: pre level and change vs own pre (x1e3), per animal. "
                   "pos = spout-position unique R&sup2; without FR; pos_fr = with FR (mode A); fr_gain = fr:full &minus; full.</p>"
                   + table(pd.concat(rows).round(1), nd=1, sign=False))
    return "".join(out)


def musall_table() -> str:
    Q = ["full", "full_fr", "alone_fr", "unique_fr", "unique_task", "uniquefr_task", "unique_instructed", "uniquefr_instructed",
         "unique_uninstructed", "uniquefr_uninstructed", "unique_position", "uniquefr_position"]
    out = []
    for lin, p in (("G4", G4), ("v36", V36)):
        d = pd.read_csv(p / "fr_encoding_summary" / "fr_musall_by_animal_modeA.csv")
        d = d[d.area == "cortex"]
        for w in ("lick", "cue"):
            m = d[d.window == w].groupby("epoch")[Q].mean().reindex(EP)
            out.append(f"<p class=small><b>{lin}</b>, cortex, {w} window: mean of animals (level).</p>" + table(m.round(3)))
    return "".join(out)


def sig_bins_table() -> str:
    rows = []
    g = pd.read_csv(G4 / "fr_encoding_summary" / "pixel_significance" / "sig_bins.csv")
    v = pd.read_csv(V36 / "fr_encoding_summary" / "pixel_significance" / "sig_bins.csv")
    for q in ("full_fr", "alone_uninstructed", "uniquefr_uninstructed", "uniquefr_task", "uniquefr_position", "unique_position"):
        for w in ("cue", "lick"):
            for ep in POST:
                a = g[(g.q == q) & (g.window == w) & (g.epoch == ep) & (g.px == 16)]
                b = v[(v.q == q) & (v.window == w) & (v.epoch == ep) & (v.px == 16)]
                fa = f"{int(a.up.iloc[0])} up / {int(a.down.iloc[0])} down" if len(a) else "&mdash;"
                fb = f"{int(b.up.iloc[0])} up / {int(b.down.iloc[0])} down" if len(b) else "&mdash;"
                rows.append(f"<tr><td class=t>{q}</td><td class=t>{w}</td><td class=t>{ep}</td><td>{fa}</td><td>{fb}</td></tr>")
    return ("<table class=num><thead><tr><th>quantity</th><th>window</th><th>epoch</th><th>G4 significant 16-px bins (of 514)</th>"
            "<th>v36</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def trial_peth_table() -> str:
    g = pd.read_csv(G4 / "fr_encoding_summary" / "trial_peth" / "bootstrap.csv")
    v = pd.read_csv(V36 / "fr_encoding_summary" / "trial_peth" / "bootstrap.csv")
    rows = []
    for ar in ("cortex", "SSp-m_left", "SSp-m_right"):
        for w in ("cue", "lick"):
            for q in ("uniquefr_position", "uniquefr_uninstructed"):
                for me in ("peth", "trial"):
                    for ep in ("acute", "chronic"):
                        a = g[(g.area == ar) & (g.window == w) & (g.q == q) & (g.measure == me) & (g.epoch == ep)]
                        b = v[(v.area == ar) & (v.window == w) & (v.q == q) & (v.measure == me) & (v.epoch == ep)]
                        if not len(a):
                            continue
                        pa = ", ".join(f"{x:+.3f}" for x in a[AN].iloc[0]).replace("-", "&minus;")
                        rows.append(f"<tr><td class=t>{ar}</td><td class=t>{w}</td><td class=t>{q}</td><td class=t>{me}</td><td class=t>{ep}</td>"
                                    f"<td>{ci(*a[['point', 'lo', 'hi']].iloc[0])} ({int(a.n_up.iloc[0])}/4 up)</td><td class=t>{pa}</td>"
                                    f"<td>{ci(*b[['point', 'lo', 'hi']].iloc[0]) if len(b) else '&mdash;'}</td></tr>")
    return ("<table class=num><thead><tr><th>area</th><th>window</th><th>quantity</th><th>variance</th><th>epoch</th><th>G4 change vs pre</th>"
            "<th>per animal PS92 / 93 / 94 / 95</th><th>v36</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def contra_table() -> str:
    g = pd.read_csv(G4 / "fr_encoding_summary" / "contra_orofacial" / "bootstrap_delta.csv")
    v = pd.read_csv(V36 / "fr_encoding_summary" / "contra_orofacial" / "bootstrap_delta.csv")
    gp = pd.read_csv(G4 / "fr_encoding_summary" / "contra_orofacial" / "per_animal_delta.csv")
    gl = pd.read_csv(G4 / "fr_encoding_summary" / "contra_orofacial" / "laterality_bootstrap.csv")
    vl = pd.read_csv(V36 / "fr_encoding_summary" / "contra_orofacial" / "laterality_bootstrap.csv")
    rows = []
    for ar in ("SSp-m_right", "SSp-m_left", "MOp_right", "cortex"):
        for w in ("cue", "lick"):
            for q in ("uniquefr_position", "uniquefr_task", "alone_uninstructed"):
                a = g[(g.area == ar) & (g.window == w) & (g.q == q) & (g.epoch == "chronic")]
                b = v[(v.area == ar) & (v.window == w) & (v.q == q) & (v.epoch == "chronic")]
                pa = gp[(gp.area == ar) & (gp.window == w) & (gp.q == q) & (gp.epoch == "chronic")].set_index("animal").delta.reindex(AN)
                pas = ", ".join(f"{x:+.3f}" for x in pa).replace("-", "&minus;")
                rows.append(f"<tr><td class=t>{ar}</td><td class=t>{w}</td><td class=t>{q}</td>"
                            f"<td>{ci(*a[['point', 'lo', 'hi']].iloc[0])} p<sub>bonf8</sub> {a.p_bonf8.iloc[0]:.3f}</td><td class=t>{pas}</td>"
                            f"<td>{ci(*b[['point', 'lo', 'hi']].iloc[0])} p<sub>bonf8</sub> {b.p_bonf8.iloc[0]:.3f}</td></tr>")
    lat = []
    for w in ("cue", "lick"):
        for q in ("uniquefr_position", "uniquefr_task"):
            for ep in ("acute", "chronic"):
                a = gl[(gl.area == "SSp-m_contra-ipsi") & (gl.window == w) & (gl.q == q) & (gl.epoch == ep)]
                b = vl[(vl.area == "SSp-m_contra-ipsi") & (vl.window == w) & (vl.q == q) & (vl.epoch == ep)]
                lat.append(f"<tr><td class=t>SSp-m contra &minus; ipsi</td><td class=t>{w}</td><td class=t>{q}</td><td class=t>{ep}</td>"
                           f"<td>{ci(*a[['point', 'lo', 'hi']].iloc[0])}</td><td>{ci(*b[['point', 'lo', 'hi']].iloc[0])}</td></tr>")
    return ("<p class=small>Chronic minus pre, a priori areas; nested bootstrap, Bonferroni over 8 area x hemisphere cells.</p>"
            "<table class=num><thead><tr><th>area</th><th>window</th><th>quantity</th><th>G4 chronic &minus; pre</th><th>per animal PS92 / 93 / 94 / 95</th><th>v36</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table><p class=small>Laterality (contra minus ipsi SSp-m), epoch minus pre.</p>"
            "<table class=num><thead><tr><th></th><th>window</th><th>quantity</th><th>epoch</th><th>G4</th><th>v36</th></tr></thead><tbody>" + "".join(lat) + "</tbody></table>")


def mechanism_table() -> str:
    g = pd.read_csv(G4 / "fr_encoding_summary" / "direction_mechanism" / "bootstrap.csv")
    v = pd.read_csv(V36 / "fr_encoding_summary" / "direction_mechanism" / "bootstrap.csv")
    rows = []
    for m in ("eta2_side", "eta2_tongue", "sd_side", "r_side_tongue", "r_SSp-m_left_side", "r_SSp-m_right_side", "r_SSp-m_c-i_side", "r_SSp-m_c-i_dist"):
        for ep in POST:
            a = g[(g.metric == m) & (g.epoch == ep)]
            b = v[(v.metric == m) & (v.epoch == ep)]
            if not len(a):
                continue
            nd = 1 if m == "sd_side" else 3
            rows.append(f"<tr><td class=t>{m}</td><td class=t>{ep}</td><td>{fmt(a.pre.iloc[0], nd)}</td><td>{ci(*a[['point', 'lo', 'hi']].iloc[0], nd=nd)} ({int(a.n_down.iloc[0])}/4 down)</td>"
                        f"<td>{ci(*b[['point', 'lo', 'hi']].iloc[0], nd=nd) if len(b) else '&mdash;'}</td></tr>")
    return ("<table class=num><thead><tr><th>metric</th><th>epoch</th><th>pre level (G4)</th><th>G4 change vs pre</th><th>v36</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def direction_table() -> str:
    p = G4 / "fr_encoding_summary" / "direction" / "bootstrap.csv"
    if not p.exists():
        return "<p class=small>fr_direction bootstrap not available at build time.</p>"
    g = pd.read_csv(p)
    v = pd.read_csv(V36 / "fr_encoding_summary" / "direction" / "bootstrap.csv")
    rows = []
    for w in ("cue", "lick"):
        for ar in ("cortex", "SSp-m_left", "SSp-m_right", "tjM1_right", "MOp_rest_right"):
            for q in ("gain_B", "gain_dist", "gain_both", "pos_B", "pos_both"):
                for ep in ("acute", "chronic"):
                    a = g[(g.window == w) & (g.area == ar) & (g.q == q) & (g.epoch == ep)]
                    b = v[(v.window == w) & (v.area == ar) & (v.q == q) & (v.epoch == ep)]
                    if not len(a):
                        continue
                    rows.append(f"<tr><td class=t>{w}</td><td class=t>{ar}</td><td class=t>{q}</td><td class=t>{ep}</td>"
                                f"<td>{ci(*a[['point', 'lo', 'hi']].iloc[0])} ({int(a.n_up.iloc[0])}/4 up)</td>"
                                f"<td>{ci(*b[['point', 'lo', 'hi']].iloc[0]) if len(b) else '&mdash;'}</td></tr>")
    return ("<p class=small>gain_B = what directionless FaceRhythm adds (fr:full &minus; full, mode B); gain_dist / gain_both = what the "
            "distance / side+distance factors add beyond B (D &minus; B, A &minus; B); pos_B / pos_both = spout-position unique R&sup2; with "
            "FR-B / FR-A. Epoch minus pre, 95% nested CI (uncorrected).</p>"
            "<table class=num><thead><tr><th>window</th><th>area</th><th>quantity</th><th>epoch</th><th>G4</th><th>v36</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def decomp_tables() -> str:
    out = []
    for lin, d in (("G4", PAGE / "tables" / "decomp_G4"), ("v36", PAGE / "tables" / "decomp_v36")):
        t = pd.read_csv(d / "fig17_1_table.csv")
        t = t[t.part.isin(["measured", "predicted", "lick part", "target part", "uninstructed", "residual"])]
        t["animals with the sign of the mean"] = t.n_same_sign.astype(str) + "/" + t.n.astype(str)
        t = t.set_index(["region", "epoch", "part"])[["mean_of_pre", *AN, "animals with the sign of the mean"]]
        out.append(f"<p class=small><b>{lin}</b>: SSp-m far contra, lick window; each part's change from the animal's own pre, divided by that animal's pre response (mean of animals; per animal).</p>"
                   + "<table class=num><thead><tr><th>region</th><th>epoch</th><th>part</th><th>mean of animals</th>" + "".join(f"<th>{a}</th>" for a in AN) + "<th>same sign</th></tr></thead><tbody>"
                   + "".join(f"<tr><td class=t>{r}</td><td class=t>{e}</td><td class=t>{p}</td><td>{fmt(row.mean_of_pre, 3, True)}</td>" + "".join(f"<td>{fmt(row[a], 3, True)}</td>" for a in AN) + f"<td>{row.iloc[-1]}</td></tr>"
                             for (r, e, p), row in t.iterrows()) + "</tbody></table>")
    return "".join(out)


def variance_table() -> str:
    g = pd.read_csv(G4 / "variance_maps" / "sig.csv")
    v = pd.read_csv(V36 / "variance_maps" / "sig.csv")
    rows = []
    for w in ("lick", "cue"):
        for ep in POST:
            a = g[(g.window == w) & (g.epoch == ep) & (g.px == 16)].iloc[0]
            b = v[(v.window == w) & (v.epoch == ep) & (v.px == 16)].iloc[0]
            rows.append(f"<tr><td class=t>{w}</td><td class=t>{ep}</td><td>{int(a.up)} up / {int(a.down)} down of {int(a.n_bins)}</td><td>{int(b.up)} up / {int(b.down)} down</td></tr>")
    return "<table class=num><thead><tr><th>window</th><th>epoch</th><th>G4 significant 16-px bins</th><th>v36</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


# ----------------------------------------------------------------------------------------------- figures
def fig_block(num: str, title: str, reading: str, img: str, inputs: dict, extra: str = "", klass="a", old_vs_new: str = "",
              note: str = "") -> str:
    body = prov({**inputs})
    if old_vs_new:
        body += f"<h4>Numbers (per animal first, then pooled); old lineage = v36 fits read by the same reader</h4>{old_vs_new}"
    imgtag = f'<a href="{img}" target=_blank><img src="{img}" alt="{esc(title)}" loading=lazy></a>' if img else "<p class=warn>(no image)</p>"
    kl = {"a": "class (a): cross-validated variance explained / unique variance (drop-one)",
          "b": "class (b): kernel shapes, betas or predicted amplitudes compared across epochs",
          "n": "model-free (no encoder weights)"}[klass]
    return (f'<figure id="fig-{num.replace(".", "-")}"><div class=ftitle><span class=fnum>Fig {num}</span>{esc(title)}</div>'
            f"{imgtag}<figcaption><span class=reading><b class=h>Reading.</b> {reading}</span>"
            f"<span class=klass>{kl}</span>{('<span class=note>' + note + '</span>') if note else ''}"
            f"{details('Inputs &amp; settings', body)}{extra}</figcaption></figure>")


def main() -> int:
    PAGE.mkdir(parents=True, exist_ok=True)
    ct_per_animal, ct_pooled = contest_tables()
    fr_sum = fr_summary_table()
    enc_std = {**LINEAGE}
    base_stats = "each animal's epoch minus its own pre, mean of animals, 95% nested bootstrap CI (animals &rarr; sessions, analysis_kit.boot_delta, seed 0); * = CI excludes 0; markers = animals (PS92 o, PS93 square, PS94 triangle, PS95 diamond)"
    figs = []

    # ---- 1.6 / 1.13 / 1.14 / 1.15
    g4p = pooled_numbers(OUT / "figs" / "pooled_numbers_1p_10p.txt")
    v36p = pooled_numbers(PAGE / "v36_same_reader" / "pooled_numbers_1p_10p.txt")
    bins_prov = {**enc_std, "Tags": "bins_w1 (width 1; Fig 1.14 / 1.15 also bins_w05, bins_w2): ONLY=nokin:, BINS=1, "
                 "movement_first order; the reader's folder name says licks-combined (v36 naming), the fits read are the four-camera ones",
                 "Window": "lick: frames within 0.2 s of a fused lick peak; cue: cue to cue + 1 s; all frames of attempted trials inside the working period",
                 "Readout": "unique dR&sup2; of the target-bin kernels (nokin:u:tgtbin = full &minus; without target bins) and of the executed-bin kernels (nokin:u:exbin), cortex, held-out (block CV); contest = target &minus; executed",
                 "Statistics": base_stats, "Script": "scripts/combined_encoding_figures.py (contest_main: region_measures.csv) -> scripts/rebuild_figs/fig_pooled.py (1p / 10p blocks, via scripts/g4figs_1125/g4_pooled.py), fig_goal_summary.py --only fig1, fig_softmax.py",
                 "Caption footer": "the PNG footer sentence (trial_set_text) is the reader's stock wording and still says cam4 licks; the fits it read are the fused four-camera ones (bundle env above)"}
    figs.append(fig_block("1.6", "Binned refit contest: pooled",
        f"Target &minus; executed still rises chronically in both windows (lick {ci(*g4p['1p contest lick chronic'], 4)}, cue {ci(*g4p['1p contest cue chronic'], 4)}; v36 {ci(*v36p['1p contest lick chronic'], 4)} / {ci(*v36p['1p contest cue chronic'], 4)}); "
        f"new: the subacute lick contest is now significant ({ci(*g4p['1p contest lick subacute'], 4)}, was {ci(*v36p['1p contest lick subacute'], 4)}); acute lick stays non-significant ({ci(*g4p['1p contest lick acute'], 4)}, PS93 the other way).",
        copy_fig(OUT / "figs" / "fig1p_contest_pooled.png", "fig1p_contest_pooled_g4.png"), bins_prov, klass="a",
        old_vs_new=ct_pooled + ct_per_animal,
        note="Conclusion unchanged (chronic target lead); one added star (subacute lick)."))
    figs.append(fig_block("1.13", "Binned refit contest: per session",
        "Per-session target-bin and executed-bin unique dR&sup2; (cortex, width 1): lines = epoch means, dots = sessions; the acute executed-bin drop and the chronic target-bin rise are visible per session.",
        copy_fig(OUT / "figs" / "fig1_binned_contest.png", "fig1_binned_contest_g4.png"), bins_prov, klass="a",
        note="Descriptive; the numbers are the per-animal tables under Fig 1.6."))
    figs.append(fig_block("1.14", "Refit contest across softmax widths: per session",
        "Widths 0.5 / 1 / 2 within each epoch; executed-bin unique R&sup2; gains as the bins widen, as before.",
        copy_fig(OUT / "figs" / "fig10_softmax_width.png", "fig10_softmax_width_g4.png"), bins_prov, klass="a"))
    w_rows = "".join(f"<tr><td class=t>{w} width {bw} {ep}</td><td>{ci(*g4p[f'10p {w} width {bw} {ep}'], 4)}</td><td>{ci(*v36p[f'10p {w} width {bw} {ep}'], 4)}</td></tr>"
                     for w in ("lick", "cue") for bw in ("0.5", "1", "2") for ep in POST)
    figs.append(fig_block("1.15", "Refit contest across softmax widths: pooled",
        f"Chronic target lead holds at every width in the lick window (0.5 / 1 / 2: {fmt(g4p['10p lick width 0.5 chronic'][0], 4, True)} / {fmt(g4p['10p lick width 1 chronic'][0], 4, True)} / {fmt(g4p['10p lick width 2 chronic'][0], 4, True)}, all *) and at widths 0.5 and 1 in the cue window, but cue width 2 is no longer significant ({ci(*g4p['10p cue width 2 chronic'], 4)}, v36 {ci(*v36p['10p cue width 2 chronic'], 4)}); acute now reaches significance at width 2 in both windows (lick {ci(*g4p['10p lick width 2 acute'], 4)}, cue {ci(*g4p['10p cue width 2 acute'], 4)}).",
        copy_fig(OUT / "figs" / "fig10p_width_pooled.png", "fig10p_width_pooled_g4.png"), bins_prov, klass="a",
        old_vs_new="<table class=num><thead><tr><th>readout</th><th>G4</th><th>v36 (same reader)</th></tr></thead><tbody>" + w_rows + "</tbody></table>",
        note="CHANGED: the v36 caption said chronic is significant at every width in both windows and acute never pooled; on G4 cue width 2 chronic loses its star and acute width 2 gains one in both windows."))

    # ---- 17.1 / 17.2 (per-session analogue)
    dec_prov = {**enc_std, "Tag": "main (thr_push, full model with LICKS_ALL=split); the page's Figs 17.1 / 17.2 read the FROZEN pre-stroke frame (lane E, wave-1 frame A); this is the PER-SESSION analogue from the same G4 fits",
                "Window": "lick-aligned PETH (peth_lick: aligned on each trial's first post-cue lick): evoked = 0 to +1 s minus &minus;0.3 to &minus;0.1 s; far-contra trials with a post-cue lick; SSp-m ipsi / contra (image left = ipsilesional)",
                "Parts": "full = task + instructed (licks, executed angle, tongue, jaw) + uninstructed (nose, running, spontaneous licks) + intercept; pos_part = position-specific part of task; residual = data &minus; full",
                "Statistics": "per animal: epoch mean over sessions minus own August pre; pooled = mean of animals; number = animals sharing the sign of the mean (no bootstrap: one value per animal and epoch)",
                "Script": "scripts/lp_encoder_decompose.py --folder movement_first_exec-polar_licks-combined --align lick -> scripts/g4figs_1125/g4_decomp_fig.py",
                "File": "tables/decomp_G4/fig17_1_table.csv, fig17_2_table.csv (v36 twin in tables/decomp_v36)"}
    figs.append(fig_block("17.1", "Encoder parts: pooled change (per-session fits)",
        "Acute: the measured far-contra SSp-m drop (&minus;0.49 ipsi / &minus;0.25 contra of the pre response) is carried by the predicted lick part (&minus;0.56 / &minus;0.35, 4/4 animals), not by the target part (~0); chronic: the measured rise (+0.26 / +0.46, 4/4) is again mostly the lick part (+0.24 / +0.40, 4/4) with a small positive residual (+0.04 / +0.06, 4/4) and no chronic residual of the size the frozen-frame version shows (+0.32 / +0.43).",
        "tables/decomp_G4/fig17_1_decomp_pooled_persession.png", dec_prov, klass="b", old_vs_new=decomp_tables(),
        note="Per-session fits re-fit the kernels in every epoch, so the residual here is the within-epoch misfit, not the frozen-frame residual of the page's 17.1; same as v36 per-session (lick part &minus;0.64 / &minus;0.49 acute, residual +0.04 / +0.07 chronic). Penalty: class (b); see the penalty section (the lick-group penalty is stable across epochs except PS95, whose pre median is 10^0 against 10^3&ndash;10^4 after the stroke)."))
    figs.append(fig_block("17.2", "SSp-m far-contra change by part (per-session fits)",
        "Bars = change from pre per part (dF/F x1e3), lick window, ipsilesional and contralesional SSp-m; markers = animals.",
        "tables/decomp_G4/fig17_2_decomp_epochs_persession.png", dec_prov, klass="b"))

    # ---- 21.x
    fr_prov = {**enc_std, "Tags": "main (no FR) and fr_A (FR=master_pre_v1, groups master_v1:A = lick amount + side + distance + other); 21.1 also fr_B / fr_S / fr_D; |full(main) &minus; full(fr_A)| max 1e-5 (same base model)",
               "Window": "ENL = pre-cue window, cue = cue to cue + 1 s, lick = frames within 0.2 s of a fused lick peak",
               "Statistics": base_stats, "Script": "scripts/fr_encoding_summary.py", "File": "fr_encoding_summary/pooled_changes.txt, fr_summary_by_animal.csv"}
    figs.append(fig_block("21.1", "FaceRhythm added R&sup2;: cortex",
        "What FaceRhythm adds to the full model now RISES after the stroke (cortex cue chronic +0.049 [+0.004, +0.102], lick acute +0.040* and chronic +0.035*; v36: ~0), while the chronic rise of spout-position unique R&sup2; without FR shrinks and loses its star (cue +0.011 [&minus;0.005, +0.028], was +0.046*; lick +0.016 [&minus;0.004, +0.036], was +0.026*); position | FR chronic cue +0.010 [+0.000, +0.020] (was +0.014*).",
        copy_fig(G4 / "fr_encoding_summary" / "figures" / "fr_summary_cortex.png", "fig16_fr_cortex_g4.png"), fr_prov, klass="a",
        old_vs_new=fr_sum + fr_per_animal("cortex", "cue") + fr_per_animal("cortex", "lick"),
        note="CHANGED: with 3-D pose the chronic cortex-wide position rise without FR is no longer significant (PS94 / PS95 no longer rise: cue +31 / +22 / &minus;8 / &minus;1 x1e3 for PS92 / 93 / 94 / 95), and FR's added R&sup2; grows chronically (PS92 +108, PS95 +77 x1e3): the 3-D movement terms explain less of the chronic face-related variance than the 2-D ones did."))
    figs.append(fig_block("21.5", "Ipsilesional SSp-m with FaceRhythm",
        "The acute loss of position coding in ipsilesional SSp-m is still there with FR (cue &minus;0.014 [&minus;0.024, &minus;0.004], lick &minus;0.012 [&minus;0.021, &minus;0.001]) and about half its v36 size (&minus;0.026 / &minus;0.022); without FR cue &minus;0.017* (v36 &minus;0.033*), lick &minus;0.014 [&minus;0.023, +0.000] (v36 &minus;0.028*).",
        copy_fig(G4 / "fr_encoding_summary" / "figures" / "fr_summary_SSp-m_left.png", "fig16_fr_SSp-m_left_g4.png"), fr_prov, klass="a",
        old_vs_new=fr_per_animal("SSp-m_left", "cue"), note="Direction unchanged; effect smaller."))
    figs.append(fig_block("21.6", "Contralesional SSp-m with FaceRhythm",
        "Chronic position unique R&sup2; with FR rises in contralesional SSp-m (cue +0.022 [+0.010, +0.035], lick +0.024 [+0.014, +0.036]; v36 +0.027* / +0.020*), while the rise WITHOUT FR is no longer significant (cue +0.008, lick +0.017 [&minus;0.004, +0.037]; v36 +0.030* / +0.026*).",
        copy_fig(G4 / "fr_encoding_summary" / "figures" / "fr_summary_SSp-m_right.png", "fig16_fr_SSp-m_right_g4.png"), fr_prov, klass="a",
        old_vs_new=fr_per_animal("SSp-m_right", "cue") + fr_per_animal("SSp-m_right", "lick")))
    mus_prov = {**fr_prov, "Readout": "alone = single-group model R&sup2;; unique = full &minus; drop-one; task-aligned (grey) = alone &minus; unique; with and without FR (mode A)",
                "Script": "scripts/fr_musall.py --mode A", "File": "fr_encoding_summary/fr_musall_by_animal_modeA.csv, fr_musall_pixel_maps_modeA.npz"}
    figs.append(fig_block("21.7", "Alone / unique R&sup2;: cue window",
        "Same layout on the four-camera fits: FR alone 0.35 (pre), unique 0.055 (v36 0.13); position unique 0.016 &rarr; 0.017 with FR pre, chronic 0.027 &rarr; 0.027; the 3-D uninstructed group now takes 0.40 alone / 0.18 unique (v36 0.25 / 0.10), so FaceRhythm's unique share halves.",
        copy_fig(G4 / "fr_encoding_summary" / "figures" / "fr_musall_cortex_cue_modeA.png", "fr_musall_cortex_cue_modeA_g4.png"), mus_prov, klass="a",
        old_vs_new=musall_table(), note="CHANGED levels: full R&sup2; 0.42 &rarr; 0.55 (pre); FR unique 0.13 &rarr; 0.06; position unique 0.036 &rarr; 0.016 (cue) / 0.046 &rarr; 0.021 (lick). The G4 design carries 3-D tongue / jaw / nose and the fused licks, which absorb variance the v36 model left to FaceRhythm and to position."))
    figs.append(fig_block("21.8", "Alone / unique R&sup2;: lick window",
        "Lick window: FaceRhythm unique 0.059 pre, 0.10 acute (v36 0.13 / 0.15); uninstructed unique 0.41 pre &rarr; 0.26 acute (v36 0.28 &rarr; 0.16); task unique 0.026 &rarr; 0.054 chronic (v36 0.067 &rarr; 0.110); position unique 0.021 &rarr; 0.037 chronic without FR (v36 0.046 &rarr; 0.073), 0.018 &rarr; 0.026 with FR (v36 0.027 &rarr; 0.032).",
        copy_fig(G4 / "fr_encoding_summary" / "figures" / "fr_musall_cortex_lick_modeA.png", "fr_musall_cortex_lick_modeA_g4.png"), mus_prov, klass="a"))
    figs.append(fig_block("21.9", "Alone / unique pixel maps: cue window",
        "Musall-style pixel R&sup2; maps with FaceRhythm, per-session fits, cue window; same layout as the page (rows = pre level, acute / chronic minus pre).",
        copy_fig(G4 / "fr_encoding_summary" / "figures" / "fr_pixel_maps_cue_modeA.png", "fr_pixel_maps_cue_modeA_g4.png"), mus_prov, klass="a"))
    figs.append(fig_block("21.10", "Alone / unique pixel maps: lick window",
        "Lick window pixel maps; descriptive (the test is Figs 21.2 / 21.3).",
        copy_fig(G4 / "fr_encoding_summary" / "figures" / "fr_pixel_maps_lick_modeA.png", "fr_pixel_maps_lick_modeA_g4.png"), mus_prov, klass="a"))
    sig_prov = {**mus_prov, "Statistics": "epoch minus pre per 16-px (and 32-px) block, mean of animals; nested bootstrap animals &rarr; sessions, Bonferroni over blocks (514 at 16 px); contours = significant blocks",
                "Script": "scripts/fr_pixel_significance.py --mode A --jobs 1; overlay scripts/fr_pixel_maps_sig.py", "File": "fr_encoding_summary/pixel_significance/sig_bins.csv"}
    sig_img_cue = copy_fig(G4 / "fr_encoding_summary" / "figures" / "fr_pixel_maps_sig_cue_modeA.png", "fr_pixel_sig_cue_g4.png")
    sig_img_lick = copy_fig(G4 / "fr_encoding_summary" / "figures" / "fr_pixel_maps_sig_lick_modeA.png", "fr_pixel_sig_lick_g4.png")
    figs.append(fig_block("21.2", "Pixel R&sup2; significance: cue window",
        "On the four-camera fits the chronic position rise is nearly absent at the pixel level even WITHOUT FaceRhythm (0 of 514 blocks up at 16 px, v36 222) and with FR 7 blocks (v36 26); task unique | FR chronic up in 100 blocks (v36 16); uninstructed unique | FR is DOWN in every epoch (342 / 113 / 207 blocks; v36 314 / 5 / 171).",
        sig_img_cue, sig_prov, klass="a", old_vs_new=sig_bins_table(),
        note="CHANGED: the v36 reading was 'chronic position rise largely absent WITH FaceRhythm' (99 blocks without, 2 with); on G4 the rise is absent without FR too, so FaceRhythm is no longer what removes it."))
    figs.append(fig_block("21.3", "Pixel R&sup2; significance: lick window",
        "Lick window: full R&sup2; up in 55 blocks chronically (v36 216); uninstructed unique | FR down broadly in every epoch (262 / 93 / 264); position unique without FR chronic up in only 5 blocks (v36 139) and with FR 10 (v36 0); task unique | FR chronic up 57 (v36 16).",
        sig_img_lick, sig_prov, klass="a", note="CHANGED as 21.2: the broad chronic full-R&sup2; and position rises of v36 are mostly gone on the 3-D design."))
    tp_prov = {**fr_prov, "Tag": "fr_A (tr_models carried in the same run; windows cue 0&ndash;1.5 s, first lick &minus;0.1&ndash;+1.0 s)",
               "Readout": "frame error = error around each spout's mean response (trial-to-trial) + error of the mean (PETH); unique R&sup2; with FR split the same way",
               "Script": "scripts/fr_trial_peth.py", "File": "fr_encoding_summary/trial_peth/bootstrap.csv"}
    figs.append(fig_block("21.11", "Trial vs average R&sup2;: pre",
        "Pre-stroke, cortex: FaceRhythm unique and uninstructed movement explain trial-to-trial variance (0.084 and 0.24 cue; 0.107 and 0.28 lick; ~0 in the PETH), task and position unique live in the PETH (0.042 and 0.035 cue; ~0 trial-to-trial): the face and the spout code still barely compete.",
        copy_fig(G4 / "fr_encoding_summary" / "trial_peth" / "figures" / "fr_trial_peth_pre_levels.png", "fr_trial_peth_pre_g4.png"), tp_prov, klass="a"))
    figs.append(fig_block("21.4", "Trial vs average R&sup2;: chronic change",
        "Chronic: contralesional SSp-m position unique | FR rises trial-to-trial (cue +0.016 [+0.003, +0.029], 3/4; lick +0.016 [&minus;0.000, +0.030]) and in the PETH (lick +0.011 [+0.002, +0.023], 4/4; cue +0.009 n.s.); the v36 acute 'PETH loss' of position is much weaker here (cortex lick PETH &minus;0.013 [&minus;0.043, +0.007], was &minus;0.036*; ipsilesional SSp-m cue PETH &minus;0.002, was &minus;0.013*), while uninstructed unique still falls trial-to-trial acutely (cortex &minus;0.12, 4/4).",
        copy_fig(G4 / "fr_encoding_summary" / "trial_peth" / "figures" / "fr_trial_peth_chronic.png", "fr_trial_peth_chronic_g4.png"), tp_prov, klass="a",
        old_vs_new=trial_peth_table(), note="CHANGED: the acute spout-coding loss is no longer a clear PETH loss; the chronic contralesional SSp-m rise is now significant trial-to-trial (cue) and in the PETH (lick)."))
    var_prov = {**enc_std, "Tag": "main (data only: the stored TOTAL|window maps = SST per frame per 2x2-px block)", "Statistics": "log2(epoch / pre) per animal, nested bootstrap, Bonferroni over blocks",
                "Script": "scripts/variance_maps.py --jobs 2", "File": "variance_maps/sig.csv, per_animal.npz"}
    figs.append(fig_block("21.12", "Total dF/F variance per pixel",
        "Unchanged: every significant change is an increase (lick window 109 / 201 / 220 blocks acute / subacute / chronic; v36 107 / 202 / 232), so a chronic R&sup2; change is not a denominator effect.",
        copy_fig(G4 / "variance_maps" / "figures" / "variance_maps_lick.png", "variance_maps_lick_g4.png"), var_prov, klass="n", old_vs_new=variance_table()))
    co_prov = {**fr_prov, "Statistics": "a priori area test: mean R&sup2; per area, nested bootstrap, Bonferroni over 8 area x hemisphere cells; laterality = contra minus ipsi; drift = pre late minus early per animal",
               "Script": "scripts/fr_contra_orofacial.py --mode A", "File": "fr_encoding_summary/contra_orofacial/{bootstrap_delta, laterality_bootstrap, chronic_vs_drift, per_animal_delta}.csv"}
    figs.append(fig_block("21.15", "Chronic change per mouse",
        "The chronic contralesional SSp-m rise in position unique | FR is now 4/4 animals and survives Bonferroni in both windows (cue +0.022 [+0.010, +0.035], lick +0.024 [+0.014, +0.036], p_bonf8 0.002 each; v36 p_bonf8 0.10 / 0.38), task unique | FR the same (+0.026 / +0.027); contra minus ipsi +0.017 (cue) / +0.025 (lick), 4/4; the per-animal change exceeds each animal's pre drift in PS92, PS94 and PS95 (lick: +0.014 / +0.027 / +0.035 / +0.020 vs drift &minus;0.001 / +0.027 / &minus;0.005 / &minus;0.001), not in PS93.",
        copy_fig(G4 / "fr_encoding_summary" / "contra_orofacial" / "figures" / "chronic_per_animal_lick_modeA.png", "fr_chronic_per_animal_lick_g4.png"), co_prov, klass="a",
        old_vs_new=contra_table(), note="STRONGER than v36 (carried by all four animals, not mainly PS94 / PS95)."))

    # ---- 22.x
    mech_prov = {"Inputs": f"{BUNDLE}/inputs/&lt;session&gt;.pkl (the G4 bundle's own imaging, behavior and raw FaceRhythm factors; no encoder weights); 3-D tongue from pose3d_4cam_3net_20261121_sync60",
                 **{k: v for k, v in LINEAGE.items() if k in ("Licks", "Trials", "Baseline")},
                 "Window": "cue 0&ndash;1.5 s, engaged (attempted) trials, 107 sessions", "Statistics": base_stats + "; p = bootstrap p",
                 "Script": "scripts/fr_direction_mechanism.py --bundle <bundle>/inputs --jobs 2", "File": "fr_encoding_summary/direction_mechanism/{sessions, bootstrap}.csv"}
    figs.append(fig_block("22.1", "FaceRhythm side index and coupling",
        "Unchanged (model-free): acutely the face's spout signal weakens (side eta&sup2; &minus;0.09, 3/4; tongue eta&sup2; &minus;0.10, 4/4) and FaceRhythm tracks the tongue less (+0.18); chronically spout-specificity and tracking are back, within-spout side movement is &minus;56 [&minus;95, &minus;17] (4/4) and ipsilesional SSp-m loses its coupling with the side index (+0.080 [+0.016, +0.142], 4/4 toward 0), so contra minus ipsi coupling falls (&minus;0.078 [&minus;0.113, &minus;0.042], 4/4).",
        copy_fig(G4 / "fr_encoding_summary" / "direction_mechanism" / "figures" / "fr_direction_mechanism_summary.png", "fr_direction_mechanism_summary_g4.png"), mech_prov, klass="n",
        old_vs_new=mechanism_table()))
    dir_prov = {**fr_prov, "Tags": "fr_B / fr_S / fr_D / fr_A (directionless; + side; + distance; + both)", "Readout": "what each FR grouping adds to the full model, and spout-position unique R&sup2; with each grouping; areas + pixel maps",
                "Script": "scripts/fr_direction.py --jobs 1", "File": "fr_encoding_summary/direction/{areas, bootstrap, pixel_sig}.csv"}
    dir_tab = direction_table()
    figs.append(fig_block("22.5", "Direction pixel maps: cue window", "Pre (top): direction factors explain lateral orofacial cortex and take position coding there; after the stroke the distance factors' added fit falls in SSp-m on both sides (cue acute &minus;0.011 / &minus;0.012, chronic &minus;0.012 / &minus;0.012 ipsi / contra, 0/4 up; v36 &minus;0.009 / &minus;0.011), while the DIRECTIONLESS gain (B) now rises chronically (ipsilesional SSp-m +0.024 [+0.002, +0.053], cortex +0.049 [+0.006, +0.101], 4/4; v36 n.s.); position unique with FR-B in contralesional SSp-m chronic +0.011 [&minus;0.005, +0.028] (v36 +0.018*), and what the direction factors take of position (A &minus; B) rises acutely (+0.011 [+0.002, +0.026], 4/4: they take less).",
        copy_fig(G4 / "fr_encoding_summary" / "direction" / "figures" / "fr_direction_maps_cue.png", "fr_direction_maps_cue_g4.png"), dir_prov, klass="a",
        old_vs_new=dir_tab, note="CHANGED: the v36 caption said the directionless gain does not change significantly anywhere; on G4 it rises chronically (and acutely in the lick window)."))
    figs.append(fig_block("22.6", "Direction pixel maps: lick window", "Lick window: the same distance-factor loss in SSp-m (acute &minus;0.007 / &minus;0.008, chronic &minus;0.008 / &minus;0.009, 0/4 up), the directionless gain rises (ipsilesional SSp-m chronic +0.024 [+0.004, +0.047], cortex acute +0.042* and chronic +0.035*), and position unique with FR-B rises chronically in contralesional SSp-m (+0.016 [+0.004, +0.028], 4/4; v36 +0.022, 3/4 n.s.).",
        copy_fig(G4 / "fr_encoding_summary" / "direction" / "figures" / "fr_direction_maps_lick.png", "fr_direction_maps_lick_g4.png"), dir_prov, klass="a"))

    # ---------------------------------------------------------------- page
    toc = "".join(f'<li><a href="#fig-{n.replace(".", "-")}">Fig {n}</a></li>' for n in
                  ["1.6", "1.13", "1.14", "1.15", "17.1", "17.2", "21.1", "21.5", "21.6", "21.7", "21.8", "21.9", "21.10", "21.2", "21.3", "21.11", "21.4", "21.12", "21.15", "22.1", "22.5", "22.6"])
    summary = open(Path(__file__).with_name("page_summary.html"), encoding="utf-8").read()
    style = """
:root { --bg:#fbfaf7; --fg:#22211f; --muted:#6b675f; --rule:#e2ded5; --accent:#6a3d9a; --card:#ffffff; --tag:#f1ece2; --side:#f4f1ea; --warn:#a33a2a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#17161a; --fg:#ece9e3; --muted:#a59f94; --rule:#34313a; --accent:#b799e0; --card:#1f1e23; --tag:#2a2830; --side:#1c1b20; --warn:#f08a78; color-scheme:dark } }
:root[data-theme="dark"] { --bg:#17161a; --fg:#ece9e3; --muted:#a59f94; --rule:#34313a; --accent:#b799e0; --card:#1f1e23; --tag:#2a2830; --side:#1c1b20; --warn:#f08a78; color-scheme:dark }
* { box-sizing:border-box } body { background:var(--bg); color:var(--fg); font:15px/1.55 "Source Sans 3", system-ui, sans-serif; margin:0 }
.toc { position:fixed; top:0; left:0; bottom:0; width:220px; overflow-y:auto; background:var(--side); border-right:1px solid var(--rule); padding:20px 14px; font-size:.86rem }
.toc ol { list-style:none; margin:0; padding:0 } .toc li { margin:0 0 5px } .toc a { color:var(--fg); text-decoration:none }
.main { margin-left:220px } .wrap { max-width:1120px; margin:0 auto; padding:24px 16px 64px }
@media (max-width: 760px) { .toc { display:none } .main { margin-left:0 } }
h1 { font:600 1.6rem/1.2 "Source Serif 4", Georgia, serif; margin:0 0 6px } h2 { font:600 1.15rem/1.3 "Source Serif 4", Georgia, serif; margin:28px 0 6px; border-top:1px solid var(--rule); padding-top:18px }
h4 { margin:12px 0 4px; font-size:.9rem } .sub { color:var(--muted); margin:0 0 18px; max-width:90ch } p, li { max-width:95ch }
.take { border-left:3px solid var(--accent); padding-left:12px; margin:10px 0 14px }
figure { margin:0 0 18px; background:var(--card); border:1px solid var(--rule); border-radius:4px; padding:10px; overflow-x:auto }
figure img { display:block; max-width:100%; height:auto; background:#fff } .ftitle { font:600 1.02rem/1.35 "Source Sans 3", sans-serif; margin:0 0 8px } .fnum { color:var(--muted); margin-right:6px }
figcaption { color:var(--muted); font-size:.86rem; margin-top:6px } figcaption b.h { color:var(--fg) } .reading { display:block; margin:0 0 4px; color:var(--fg) }
.klass { display:inline-block; font-size:.72rem; letter-spacing:.04em; text-transform:uppercase; background:var(--tag); padding:1px 6px; border-radius:3px; margin:2px 0 }
.note { display:block; color:var(--warn); margin:4px 0 } .capinfo { margin-top:4px } .capinfo > summary { cursor:pointer; font-size:.78rem }
.provtab { font-size:.78rem; border-collapse:collapse; margin:8px 0 4px; width:100% } .provtab th { text-align:left; padding:2px 10px 2px 0; white-space:nowrap; vertical-align:top; color:var(--muted) } .provtab td { padding:2px 0; vertical-align:top } .provtab tr + tr th, .provtab tr + tr td { border-top:1px solid var(--rule) }
table.num { border-collapse:collapse; font-size:.8rem; font-variant-numeric:tabular-nums; margin:4px 0 10px } table.num td, table.num th { border-bottom:1px solid var(--rule); padding:2px 7px; text-align:right; white-space:nowrap } table.num td.t, table.num th:first-child { text-align:left }
b.star { color:var(--accent) } .small { font-size:.82rem; color:var(--muted) } code { font-size:.85em }
"""
    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>G4 key results</title><style>{style}</style></head>
<body><nav class=toc><b>G4 key results</b><ol><li><a href="#summary">What G4 shows</a></li><li><a href="#tables">Shares, R&sup2;, penalties</a></li>{toc}<li><a href="#skipped">Skipped</a></li></ol></nav>
<div class=main><div class=wrap>
<h1>G4 per-session encoder (four-camera 3-D): key results of the Movement (Execution) Coding page</h1>
<p class=sub>Local build 2026-11-25 (CC-B helper), not published. G4 = enc_fourcam3d_20261124: combined_encoding per-session fits, 20 tags x 107 sessions, fused four-camera licks, 3-D tongue / jaw / nose, jaw rule 3d_mouth, attempted gate, working period, August-only pre, running model thr_push. Figures 1.6, 1.13&ndash;1.15, 17.1&ndash;17.2, 21.x and 22.x of the page re-rendered from these fits with the page's own readers; old lineage = the v36 collected fits read by the same reader. Pooled numbers are each animal's change from its own pre with the nested animals &rarr; sessions bootstrap; per-animal tables come first in every "Inputs &amp; settings" block.</p>
<section id=summary><h2>What G4 shows</h2>{summary}</section>
<section id=tables><h2>Shares, held-out R&sup2; and selected penalties</h2>
<h3>Held-out cortex R&sup2;, main (thr_push) vs main_spline</h3>
<p class=small>From run_records/2026-11-24/g4_running_compare_ccB1124/main_spline_107/SUMMARY.txt (median per-session cortex R&sup2;; 107 sessions).</p>
<table class=num><thead><tr><th>window</th><th>main</th><th>main_spline</th><th>spline &minus; main (median)</th><th>sessions spline higher</th></tr></thead><tbody>
<tr><td class=t>all</td><td>0.6097</td><td>0.6185</td><td>+0.0042</td><td>78/107</td></tr><tr><td class=t>enl</td><td>0.4949</td><td>0.5000</td><td>+0.0036</td><td>82/107</td></tr>
<tr><td class=t>cue</td><td>0.5561</td><td>0.5540</td><td>+0.0017</td><td>61/107</td></tr><tr><td class=t>lick</td><td>0.5539</td><td>0.5571</td><td>+0.0055</td><td>76/107</td></tr>
<tr><td class=t>cuestop</td><td>0.6399</td><td>0.6448</td><td>+0.0041</td><td>80/107</td></tr><tr><td class=t>firstlick</td><td>0.5054</td><td>0.5050</td><td>+0.0025</td><td>62/107</td></tr></tbody></table>
<p class=small>Other comparison tags (same source): nopush &minus;0.006 (lower in 104/107), pushhinge +0.0016 (90/107), precap +0.0002. thr_push stays primary (Priya 11-24).</p>
<h3>Unique target / executed / position-drop shares by epoch, main vs main_spline</h3>
<p class=small>From run_records/2026-11-24/g4_running_compare_ccB1124/shares/shares_main_vs_spline.csv (cortex, held-out R&sup2; fraction, August-only pre). Script scripts/g4figs_1125/g4_shares.py; tables/shares/.</p>
{details('Per animal and pooled, by window (lick, cue, all)', shares_tables())}
<h3>Selected ridge penalties per session (class (b) sensitivity)</h3>
<p class=small>Every session's full model picks one penalty per group (task / lick / orofacial / uninstructed) by a two-pass coordinate search over 1, 10, ..., 10^6 on held-out R&sup2; (combined_encoding.py); every non-shuffle reduced model (drop:&lt;group&gt;, nokin:&lt;set&gt;) re-searches the groups it contains. Values below are log10(penalty). Script scripts/g4figs_1125/g4_penalties.py; tables/penalties/.</p>
{details('Per animal x epoch, G4 and v36 (full model, tag main)', penalties_tables(), open_=True)}
<p class=small>Reduced models: in G4 main the drop:position model re-selects a different lick / orofacial penalty than the full model in ~54% of sessions (mean |&Delta;log10| 0.9&ndash;1.0 for lick and orofacial, 0.25 for uninstructed); drop:instructed differs in 35%. The bins_w1 contest models (nokin:tgtbin / nokin:exbin / nokin:none) keep median penalties flat across epochs (lick 3&ndash;4, task 4, orofacial 1 &rarr; 2 chronic, uninstructed 1 &rarr; 2 chronic).</p>
</section>
<section id=figs><h2>Figures</h2>{''.join(figs)}</section>
<section id=skipped><h2>Skipped and not re-rendered</h2>
<ul>
<li><b>Fig 22.2</b> (time-shifted FaceRhythm, contest_fr_lags): needs the frozen frame <code>_frA_frlag</code>, which does not exist on this lineage (no fit run here).</li>
<li><b>Figs 22.3 / 22.4</b> (direction vs directionless by area): images were cut from the page 10-29 (docs/CUT_FIGURES.md); the G4 numbers are in the table under Fig 22.5 (fr_encoding_summary/direction/bootstrap.csv).</li>
<li><b>Figs 21.13 / 21.14</b>: schematics, not data-fed.</li>
<li><b>Page Figs 17.1 / 17.2 as drawn</b> read the frozen pre-stroke frame (lane E wave-1 frame A), not G4; the blocks above are the per-session analogue from the same decomposition reader.</li>
<li>Running-model sensitivity rows (main_spline / main_thr_pushhinge) are not drawn; the shares table shows main vs spline agree.</li>
</ul></section>
<p class=small>Record: G4_KEY_RESULTS_20261125.md beside this file. Rendered outputs: {esc(str(OUT))} (readers' tables under imaging_encoding_view/&lt;run tag&gt;/...), page tables under tables/.</p>
</div></div></body></html>"""
    (PAGE / "index_g4_key_results_20261125.html").write_text(doc, encoding="utf-8")
    print("->", PAGE / "index_g4_key_results_20261125.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Quiet-period event figures redrawn under Priya's figure conventions (11-18, via 3c: "OK'd the restyle"; "use k = 4
for all"). Reads only the tables the quiet scripts already wrote (no extraction, no LMM refit) and writes the restyled
PNGs (+ svg/) and the star-change list to <k 4 run>/restyled_20261118/.

CONVENTIONS (Priya 11-03 / 11-07): every readout = change from the animal's OWN pre; open session markers (shape =
animal, color = epoch), pre sessions drawn the same way (the no-stroke spread around 0); filled marker + bar = the
animal-weighted change with its 95 % CI (`analysis_kit.epoch_change` = `boot_delta`, nested animals -> sessions);
PRIMARY "*" = that CI excludes 0 (top band, `sig_marks.star_top`); BACKUP dagger = the script's own LMM (mixed_models
.lmm_table, FDR within measure, the q its figure used for stars) q < .05, second row; "k/4" = animals whose own change
has the sign of the pooled change; no line across epochs; no top spine (`reserve_headroom`). Rates are changes in
events / min; SD / power / rate ratios of the 415 check are fold vs own pre (change of log10, drawn on a log axis).
T13 is a within-session contrast (event trials minus the same-session same-spout mean), so its star is the LEVEL test:
nested bootstrap (`boot_ci`) of the per-session excess excludes 0, per epoch; dagger = the old within-stratum
permutation p < .05.

INPUTS: base summary (pooled epochs) = the common-k-4 guard summary v2_quiet_lp_c1guard_k4_20261118 (Priya 11-18);
follow-on tables (face-regressed, 415, stillness, task windows, decay) = v2_quiet_lp_c1guard_k375_20261117/<family>/:
those scripts use their own fixed k = 4 (the folder name is the base summary's k), so they ARE the k-4 numbers.

    PYTHONPATH=<worktree>;<worktree>/scripts python scripts/cohort2_qc/quiet_restyle_figs.py [--only fr hemo ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wfield_local import analysis_kit as kit  # noqa: E402
from wfield_local import cat_points as cp  # noqa: E402
from wfield_local import sig_marks  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402
from wfield_local.orofacial_cohort import EPOCH_LINE, POS_ORDER  # noqa: E402

T = "hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk"
Q = Path(f"M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/{T}/cohort1_quiet_spikes")
BASE = Q / "v2_quiet_lp_c1guard_k4_20261118"
FOL = Q / "v2_quiet_lp_c1guard_k375_20261117"
OUT = BASE / "restyled_20261118"
EP = ["pre", "acute", "subacute", "chronic"]
POST = EP[1:]
HEMI = {"_left": " ipsilesional", "_right": " contralesional"}
DAGGER = "\u2020"
RECORDS: list = []
STATS = ("* = nested animals -> sessions bootstrap 95% CI of the change vs own pre excludes 0 (primary); "
         f"{DAGGER} = mixed model q < .05 (backup, FDR within measure); k/4 = animals changing in the pooled direction")
LOOK = ("Open markers = sessions minus their animal's own pre mean (shape = animal, color = epoch; pre sessions = the "
        "no-stroke spread); filled = animal-weighted change, 95% CI")


def hemi(c):
    for k, v in HEMI.items():
        c = c.replace(k, v)
    return c


def stars_q(q):
    return "" if not np.isfinite(q) else "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def lmm_q(Tab, **filt):
    """{epoch: q} of the epoch model terms matching filt (the q each old figure starred)."""
    m = Tab.model == "epoch" if "model" in Tab else np.ones(len(Tab), bool)
    for k, v in filt.items():
        m &= Tab[k] == v
    out = {}
    for ep in POST:
        t = Tab[m & (Tab.term == f"C(epoch)[T.{ep}]")]
        out[ep] = float(t.q.iloc[0]) if len(t) else np.nan
    return out


def change_cells(ax, S, col, rng, xpos, *, q=None, off=None, fold=False, panel="", seed=0, jitter=0.1, dx=0.28,
                 row=0, kcount=True):
    """Session dots (minus own pre mean) at xpos[ep] + the animal-weighted change and its 95 % CI at xpos[ep] + dx for
    each post epoch; * = CI excludes 0 (row `row`), dagger = q[ep] < .05 (row `row` + 1). off: analyze log10(v + off)
    (fold vs own pre, drawn as 10**change on a log axis when fold). Records one row per post epoch in RECORDS."""
    D = S[["animal", "epoch", col]].copy()
    v = D[col].astype(float)
    if off is not None:
        v = np.log10(v + off)
    D["v"] = v.replace([np.inf, -np.inf], np.nan)
    D = D.dropna(subset=["v"])
    pre = D[D.epoch == "pre"].groupby("animal").v.mean()
    D = D[D.animal.isin(pre.index)]
    D["d"] = D.v - D.animal.map(pre)
    show = (lambda x: 10 ** np.asarray(x, float)) if fold else (lambda x: np.asarray(x, float))
    for i, ep in enumerate(EP):
        g = D[D.epoch == ep]
        cp.session_dots(ax, xpos[ep], show(g.d), g.animal, EPOCH_LINE[ep], jitter=jitter, seed=seed + i)
    ec = kit.epoch_change(D, "v", rng, POST)
    for ep in POST:
        c = ec[ep]
        x = xpos[ep] + dx
        qq = (q or {}).get(ep, np.nan)
        star = bool(np.isfinite(c.lo) and (c.lo > 0 or c.hi < 0))
        same = (c.n_up if c.point > 0 else c.n_animals - c.n_up) if np.isfinite(c.point) else 0
        if np.isfinite(c.point):
            p, lo, hi = show(c.point), show(c.lo), show(c.hi)
            ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="o", color=EPOCH_LINE[ep], ms=6, lw=1.8, capsize=0,
                        mec="w", mew=0.6, zorder=5)
            if kcount:
                ax.text(x, 0.01, f"{same}/{c.n_animals}", transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=6.5, color=EPOCH_LINE[ep])
        if star:
            sig_marks.star_top(ax, x, "*", color=EPOCH_LINE[ep], row=row)
        if np.isfinite(qq) and qq < 0.05:
            sig_marks.star_top(ax, x, DAGGER, color=EPOCH_LINE[ep], row=row + 1, fontsize=10)
        RECORDS.append(dict(panel=panel, epoch=ep, scale="log10 fold" if off is not None else "difference",
                            point=c.point, lo=c.lo, hi=c.hi, k_same=same, n_animals=c.n_animals,
                            per_animal="; ".join(f"{a} {d:+.3g}" for a, d in sorted(c.per_animal.items())),
                            new_star="*" if star else "", lmm_q=qq, old_star=stars_q(qq)))
    ax.axhline(1 if fold else 0, color="0.6", lw=0.8, zorder=0)
    if fold:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v_, _: f"{v_:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    return D


def epoch_axis(ax):
    ax.set_xticks(range(len(EP)), EP, fontsize=8)
    ax.set_xlim(-0.5, len(EP) - 0.4)


XE = {ep: i for i, ep in enumerate(EP)}


def legend(fig, y=0.995, extra=()):
    hs = cp.animal_handles(["PS92", "PS93", "PS94", "PS95"])
    hs += [plt.Line2D([], [], marker="o", ls="", color=EPOCH_LINE[e], mec="w", label=e) for e in EP]
    hs += list(extra)
    fig.legend(handles=hs, loc="upper center", ncol=len(hs), frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, y))


def finish(fig, name, title, foot, top=0.9):
    sig_marks.reserve_headroom(fig)
    fig.suptitle(title, fontsize=9.5, y=0.997, va="top")
    fig.text(0.01, 0.003, foot, fontsize=7, wrap=True, va="bottom", color="0.25")
    fig.tight_layout(rect=(0, 0.06, 1, top))
    OUT.mkdir(exist_ok=True)
    save_png_svg(fig, OUT / name, dpi=110)
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------------------------------------------- families
def fig_pooled(rng):
    S = pd.read_csv(BASE / "quiet_events_sessions.csv")
    L = pd.read_csv(BASE / "lmm_v2.csv")
    cols = ["orofacial_left", "orofacial_right", "limb_left", "limb_right"]
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.4), sharey="row")
    for r, (m, lab) in enumerate((("rate", "POSITIVE events"), ("rate_both", "EITHER polarity"))):
        for j, c in enumerate(cols):
            a = ax[r, j]
            change_cells(a, S, f"{c}|{m}", rng, XE, q=lmm_q(L, measure=m, column=c), panel=f"pooled {lab} {hemi(c)}")
            epoch_axis(a)
            a.set_title(f"{lab}\n{hemi(c)}", fontsize=9)
        ax[r, 0].set_ylabel("events / min engaged quiet,\nchange vs own pre")
    legend(fig, 0.95)
    finish(fig, "quiet_events_pooled_epochs_restyled.png",
           "Quiet-period cortical events, change vs each animal's own pre (common k = 4 for orofacial and limb, Priya "
           "11-18; cam1 far-from-mouth guard)",
           "Events = height above the preceding 1.0-0.2 s > 4 x the animal's August-pre robust SD (1.4826 x MAD) in the "
           "2-s-median-detrended dF/F; quiet = no lick (DAQ + LP video, cam1 guard) / running within 1.5 s; engaged part "
           f"of the session; August pre. {LOOK}. {STATS}. Source: {BASE.name}/quiet_events_sessions.csv, lmm_v2.csv.")


def fig_fr(rng, detail=False):
    S = pd.read_csv(FOL / "fr_regressed_own" / "quiet_events_fr_sessions.csv")
    L = pd.read_csv(FOL / "fr_regressed_own" / "lmm_v2_fr.csv")
    cols = ["orofacial_left", "orofacial_right", "limb_left", "limb_right"]
    rows = (["raw_both", "res_both", "move_frac", "r2"] if detail else ["res_both"])
    ylab = {"res_both": "events / min, face regressed,\nchange vs own pre",
            "raw_both": "events / min, raw,\nchange vs own pre",
            "move_frac": "fraction of raw events with face\nmovement, change vs own pre",
            "r2": "face model CV R2,\nchange vs own pre"}
    fig, ax = plt.subplots(len(rows), 4, figsize=(15, 3.0 * len(rows) + 1.3), sharey="row", squeeze=False)
    for i, m in enumerate(rows):
        for j, c in enumerate(cols):
            a = ax[i, j]
            change_cells(a, S, f"{c}|{m}", rng, XE, q=lmm_q(L, measure=m, column=c),
                         panel=f"{'fr_detail' if detail else 'fr'} {m} {hemi(c)}")
            epoch_axis(a)
            if i == 0:
                a.set_title(hemi(c), fontsize=9)
        ax[i, 0].set_ylabel(ylab[m], fontsize=8)
    legend(fig, 0.95 if not detail else 0.975)
    chance = S.groupby("epoch").move_null.mean().reindex(EP)
    extra = (" Chance share of frames with face movement (random engaged quiet frames), mean per epoch: "
             + " / ".join(f"{v:.2f}" for v in chance) + " (pre / acute / subacute / chronic)." if detail else "")
    finish(fig, f"quiet_events_face_regressed{'_detail' if detail else ''}_restyled.png",
           ("Raw vs face-regressed quiet events, movement at raw events and face-model R2" if detail else
            "Quiet-period events per minute after regressing out face movement") + ", change vs own pre",
           "Face model = each session's own FaceRhythm factors + face-point PCs + speeds, lags +-0.3 s, ridge out of "
           "fold; events > 4 x the animal's August-pre robust SD of the raw / residual trace, positive + negative; quiet "
           f"= no lick / running within 1.5 s, engaged part. {LOOK}. {STATS}.{extra} Source: "
           f"{FOL.name}/fr_regressed_own (fixed k = 4).", top=0.9 if not detail else 0.95)


def fig_hemo(rng):
    S = pd.read_csv(FOL / "hemo415" / "quiet_hemo_sessions.csv")
    L = pd.read_csv(FOL / "hemo415" / "lmm_hemo415.csv")
    CH = {"corr470": "corrected 470 (analyzed)", "raw470": "uncorrected 470", "raw415": "415 nm (calcium-insensitive)"}
    meas = [("sd", "quiet robust SD", 0.0), ("p14", "quiet 1-4 Hz power", 0.0), ("p0510", "quiet 0.5-1 Hz power", 0.0),
            ("rate", "event rate, own-channel k = 4\n(rate + 0.2)", 0.2), ("eta", "amplitude at corrected-470\nevents "
                                                                              "(dF/F %)", None)]
    fig, ax = plt.subplots(3, len(meas), figsize=(3.4 * len(meas), 9.6), squeeze=False)
    for i, v in enumerate(CH):
        for j, (m, lab, off) in enumerate(meas):
            a = ax[i, j]
            change_cells(a, S, f"{v}|orofacial|{m}", rng, XE, q=lmm_q(L, measure=m, column=f"{v}|orofacial"),
                         off=off, fold=off is not None, panel=f"hemo {v} {m}")
            epoch_axis(a)
            if i == 0:
                a.set_title(lab + ("\n(fold vs own pre)" if off is not None else "\n(change vs own pre)"), fontsize=8)
            if j == 0:
                a.set_ylabel(CH[v], fontsize=8)
    legend(fig, 0.95)
    finish(fig, "quiet_hemo_check_restyled.png",
           "Is the post-stroke quiet-period change calcium or hemodynamic? Bilateral orofacial ROI, three channels, "
           "change vs each animal's own pre",
           "Engaged quiet frames, August pre. SD / power / rate: fold vs own pre = 10^(session log10 value - the "
           "animal's mean pre log10 value), the bootstrap on the log scale (as the mixed model); amplitude: difference "
           f"in dF/F %. {LOOK}. {STATS}. The amplitude column's random-quiet-frame null (the old gray band) is "
           f"not drawn in change units (per-session null in quiet_hemo_sessions.csv). Source: {FOL.name}/hemo415 (fixed k = 4).", top=0.92)


def fig_stillness(rng):
    S = pd.read_csv(FOL / "stillness" / "stillness_sessions.csv")
    L = pd.read_csv(FOL / "stillness" / "lmm_rate_by_bin.csv")
    bins = sorted(S.bin.unique())
    lab = list(L.drop_duplicates("binl").binl) if len(L) else [str(b) for b in bins]
    S = S.assign(share=S.minutes / S.groupby("label").minutes.transform("sum"))
    off = cp.epoch_offsets(len(EP), step=0.21)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
    for k, (val, title, ylab) in enumerate((("rate", "event rate by face stillness", "orofacial events / min,\nchange "
                                             "vs own pre (same bin)"),
                                            ("share", "time spent at each stillness level", "share of engaged-quiet "
                                             "time,\nchange vs own pre"))):
        a = ax[k]
        for b in bins:
            G = S[S.bin == b]
            xp = {ep: b + off[i] for i, ep in enumerate(EP)}
            q = lmm_q(L, binl=lab[b]) if val == "rate" else None
            change_cells(a, G, val, rng, xp, q=q, panel=f"stillness {val} bin {lab[b]}", seed=10 * b, jitter=0.03,
                         dx=0.0, kcount=False)
        for xb in np.arange(len(bins) - 1) + 0.5:
            a.axvline(xb, color="0.92", lw=0.6, zorder=0)
        a.set_xticks(range(len(bins)), lab)
        a.set_xlabel("face speed / animal's pre median (left = stillest)")
        a.set_ylabel(ylab)
        a.set_title(title + "\n(epochs left to right in each bin: pre, acute, subacute, chronic)", fontsize=9)
    legend(fig, 0.95)
    finish(fig, "quiet_stillness_restyled.png",
           "T7: is the acute rise in quiet orofacial events explained by a stiller face? Change vs own pre within each "
           "stillness bin",
           "Engaged quiet frames binned by face speed / the animal's pre median; rate = orofacial events (raw, up + "
           "down, > 4 x pre SD) per minute in that bin (bins with < 0.5 min in a session left out). Filled = the "
           "animal-weighted change at the bin, 95% CI (dots and summary share the epoch position); stars per bin and "
           f"epoch. {STATS} (no mixed model for the time share). Source: {FOL.name}/stillness.", top=0.9)


def fig_t12(rng):
    S = pd.read_csv(FOL / "task_events" / "t12_session_rates.csv")
    L = pd.read_csv(FOL / "task_events" / "t12_lmm.csv")
    A = pd.read_csv(FOL / "task_events" / "t12_animal_epoch_rates.csv")
    ws = [("precue", "PRE-CUE (spout in place, before the cue)"), ("iti", "ITI (trial end to next spout arrival)"),
          ("quiet", "QUIET (v2 rule, reference)")]
    fig, ax = plt.subplots(3, 3, figsize=(14, 12))
    for r, c in enumerate(["orofacial_left", "orofacial_right"]):
        for j, (w, wn) in enumerate(ws):
            a = ax[r, j]
            change_cells(a, S, f"{c}|{w}", rng, XE, q=lmm_q(L, measure=w, column=c), panel=f"t12 {w} {hemi(c)}")
            epoch_axis(a)
            a.set_title(f"{wn}\n{hemi(c)}", fontsize=9)
            if j == 0:
                a.set_ylabel("events / min, change vs own pre")
    for j, (w, name) in enumerate((("precue", "pre-cue"), ("iti", "ITI"))):        # per animal x epoch folds, as before
        a = ax[2, j]
        for c, fill in (("orofacial_left", "none"), ("orofacial_right", None)):
            Qf = A[(A.column == c) & (A.window == "quiet")].set_index(["animal", "epoch"]).fold_vs_pre
            Pf = A[(A.column == c) & (A.window == w)].set_index(["animal", "epoch"]).fold_vs_pre
            for (an, ep), qv in Qf.items():
                if ep == "pre" or not np.isfinite(qv) or not np.isfinite(Pf.get((an, ep), np.nan)):
                    continue
                a.scatter(qv, Pf[(an, ep)], marker=cp.animal_marker(an), s=40, lw=1.0, edgecolors=EPOCH_LINE[ep],
                          facecolors="none" if fill == "none" else EPOCH_LINE[ep])
        lim = [0.5, max(2, float(np.nanmax(A[A.window.isin(["quiet", w])].fold_vs_pre)) * 1.1)]
        a.plot(lim, lim, color="0.6", lw=0.8, ls="--")
        a.axhline(1, color="0.85", lw=0.6)
        a.axvline(1, color="0.85", lw=0.6)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("QUIET rate, fold vs the animal's pre")
        a.set_ylabel(f"{name.upper()} rate, fold vs pre")
        a.set_title(f"Per animal x epoch: {name} vs quiet change\n(open = ipsilesional, filled = contralesional;\n"
                    "dashed = equal change; no test)", fontsize=9)
    a = ax[2, 2]
    for k, (w, _short) in enumerate((("precue_notquiet", "not quiet"), ("precue_inquiet", "quiet"))):
        xp = {ep: i + (k - 0.5) * 0.42 for i, ep in enumerate(EP)}
        change_cells(a, S, f"orofacial|{w}", rng, xp, q=lmm_q(L, measure=w, column="orofacial"),
                     panel=f"t12 {w} orofacial", seed=10 * k, jitter=0.05, dx=0.13, row=0, kcount=False)
    a.set_xticks([i + (k - 0.5) * 0.42 for i in range(len(EP)) for k in range(2)],
                 [f"{ep}\n{s}" for ep in EP for s in ("not\nquiet", "quiet")], fontsize=7)
    for xb in np.arange(len(EP) - 1) + 0.5:
        a.axvline(xb, color="0.92", lw=0.6, zorder=0)
    a.set_ylabel("events / min, change vs own pre")
    a.set_title("Pre-cue events (orofacial bilateral), split by whether\nthe frame is itself a v2 QUIET frame",
                fontsize=9)
    legend(fig, 0.95)
    finish(fig, "t12_task_window_event_rates_restyled.png",
           "T12: orofacial events in TASK windows, change vs each animal's own pre",
           "v2 detector (2-s median detrended dF/F %, height > 4 x the animal's August-pre robust SD, either polarity); "
           "task masks exclude frames within 0.5 s of a contact lick / running / swapped pair; engaged part of the "
           f"session. {LOOK}. {STATS} (mixed model on log10(rate + 0.2)). Source: {FOL.name}/task_events.", top=0.92)


def fig_decay(rng):
    import quiet_eta_fr as E                              # TRACE_COLS / GROUPS, the trace definition of the old figure
    Tb = pd.read_csv(FOL / "eta_fr" / "decay_sessions.csv")
    L = pd.read_csv(FOL / "eta_fr" / "lmm_decay.csv")
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    files = sorted((FOL / "eta_fr" / "sessions").glob("*.npz"))
    ep_of = Tb.drop_duplicates("label").set_index("label").epoch.to_dict()
    for r, g in enumerate(E.GROUPS):
        a = ax[r, 0]
        per = {ep: {} for ep in EP}
        lags = None
        for f in files:
            ep = ep_of.get(f.stem)
            if ep is None:
                continue
            with np.load(f) as z:
                if f"etatr_{g}_pos" not in z.files or int(z[f"n_{g}_pos"]) < 5:
                    continue
                lags = z["lags_s"]
                per[ep].setdefault(f.stem[:4], []).append(z[f"etatr_{g}_pos"][E.TRACE_COLS.index(g)])
        for ep in EP:
            ms = []
            for _an, Z in sorted(per[ep].items()):
                m = np.nanmean(Z, 0)
                ms.append(m / np.nanmax(m[np.abs(lags) < 0.15]))
            if ms:
                a.plot(lags, np.mean(ms, 0), color=EPOCH_LINE[ep], lw=2, label=f"{ep} ({len(ms)} an.)")
        a.axhline(0.5, color="0.6", lw=0.6, ls="--")
        a.axvline(0, color="0.6", lw=0.6)
        a.set_xlim(-0.6, 1.5)
        a.set_xlabel("s from event peak")
        a.set_title(f"{g} UP events, residual trace, normalized to peak\n(mean of animals)", fontsize=8.5)
        a.legend(fontsize=7, frameon=False)
        for c, (m, lab) in enumerate((("decay_half_s", "peak -> half decay (s)"), ("rel_at_0p5", "value at +0.5 s / "
                                                                                                 "peak")), 1):
            a = ax[r, c]
            G = Tb[(Tb.group == g) & (Tb.kind == "pos")]
            change_cells(a, G, m, rng, XE, q=lmm_q(L, measure=m, group=g), panel=f"decay {g} {m}")
            epoch_axis(a)
            a.set_title(f"{g} UP: {lab}, change vs own pre", fontsize=8.5)
    legend(fig, 0.95)
    finish(fig, "quiet_eta_decay_fr_restyled.png",
           "Do acute events decay faster? Movement-corrected (face-regressed) event-triggered residual traces",
           "Face-regressed events (> 4 x the animal's August-pre robust SD of the residual trace), engaged quiet; per "
           "session (>= 5 UP events) from the event-triggered mean residual trace: time from the peak to half its height "
           f"and the value at +0.5 s / peak. {LOOK}. {STATS}. Source: {FOL.name}/eta_fr.", top=0.92)


def fig_t13(rng):
    import quiet_task_events as QT                       # outcomes() = the trial subsets / strata of T13
    Tr = pd.read_csv(FOL / "task_events" / "t13_trials.csv.gz")
    R = pd.read_csv(FOL / "task_events" / "t13_effects.csv")
    RS = pd.read_csv(FOL / "task_events" / "t13_by_spout.csv")
    O = QT.outcomes(Tr)
    panels = [("farcontra_fail", "far-contra FAILED attempt\nexcess P(fail) on event trials"),
              ("farcontra_incomplete_vs_hit", "far-contra INCOMPLETE vs hit\nexcess P(incomplete)"),
              ("allspout_miss", "all spouts, MISS vs hit (spout-matched)\nexcess P(miss)"),
              ("p_true_enl_event", "frozen pre-stroke decoder p(true spout), 2-s pre-cue\nexcess p_true, event in that "
                                   "window")]
    fig, ax = plt.subplots(2, 3, figsize=(16, 10.5))
    sess_rows = []
    for k, (name, title) in enumerate(panels):
        B, flag, _ = O[name]
        rows = []
        for lab_, G in B.groupby("label"):                # per-session excess over the session's STRICT strata
            r = QT.excess(G.y, G[flag], G.strict, n_perm=0)
            rows.append(dict(label=lab_, animal=lab_[:4], epoch=G.epoch.iloc[0], excess=r["excess"],
                             n_event=r["n_event"]))
        P = pd.DataFrame(rows)
        sess_rows.append(P.assign(outcome=name))
        old = R[(R.outcome == name) & (R.animal == "ALL")].set_index("epoch").p_strict
        level_cells(ax.flat[k], P, "excess", rng, {ep: i for i, ep in enumerate(EP)}, old, f"t13 {name}")
        epoch_axis(ax.flat[k])
        ax.flat[k].set_title(title, fontsize=9)
        ax.flat[k].set_ylabel("excess, event trials minus\nsame-session strata mean")
    off = cp.epoch_offsets(len(EP), step=0.2)
    for k, (name, title) in enumerate((("allspout_miss", "excess P(miss) by spout"),
                                       ("p_true_enl_event", "excess p_true by spout"))):
        a = ax.flat[4 + k]
        B, flag, _ = O[name]
        rows = []
        for (lab_, sp), G in B.groupby(["label", "pos"]):  # per session x spout (the old by-spout stratum)
            r = QT.excess(G.y, G[flag], G.stratum, n_perm=0)
            rows.append(dict(label=lab_, animal=lab_[:4], epoch=G.epoch.iloc[0], spout=sp, excess=r["excess"]))
        P = pd.DataFrame(rows)
        for b, sp in enumerate(POS_ORDER):
            old = RS[(RS.outcome == name) & (RS.spout == sp)].set_index("epoch").p
            level_cells(a, P[P.spout == sp], "excess", rng, {ep: b + off[i] for i, ep in enumerate(EP)}, old,
                        f"t13 {name} {sp}", jitter=0.03, dx=0.0, rows_by_epoch=True, seed=10 * b)
        for xb in np.arange(len(POS_ORDER) - 1) + 0.5:
            a.axvline(xb, color="0.92", lw=0.6, zorder=0)
        a.set_xticks(range(len(POS_ORDER)), [p.replace("_", "\n") for p in POS_ORDER], fontsize=8)
        a.set_title(f"{title}, sessions\n(stars: one row per epoch)", fontsize=9)
    pd.concat(sess_rows).to_csv(OUT / "t13_session_excess.csv", index=False)
    legend(fig, 0.95)
    finish(fig, "t13_precue_event_vs_outcome_restyled.png",
           "T13: does a pre-cue orofacial event predict the trial's outcome or its pre-cue position code?",
           "Event = any orofacial (ipsi or contra) event > 4 x pre SD, either polarity, in the trial's clean pre-cue "
           "window (>= 1 s clean, no lick / run within 0.5 s); engaged (canonical) trials. Per session: mean over event "
           "trials of (outcome - its stratum mean), STRICT strata = spout x trial-order quartile x clean-time bin "
           "(by-spout panels: spout only); sessions with < 3 event trials left out. Open markers = sessions (shape = "
           "animal, color = epoch); filled = pooled mean, 95% CI (boot_ci, nested animals -> sessions). * = CI excludes "
           f"0 (primary; a within-session contrast, so tested against 0 in each epoch); {DAGGER} = the old pooled "
           "within-stratum permutation p < .05 (backup). PS93_0813 excluded (no video): 31 pre sessions. Source: "
           f"{FOL.name}/task_events/t13_trials.csv.gz (p_true = production-key attempt_coding tables).", top=0.92)


def level_cells(ax, P, col, rng, xpos, old_p, panel, *, jitter=0.1, dx=0.28, rows_by_epoch=False, seed=0):
    """T13: session values as open markers + pooled level with 95 % CI (boot_ci); * = CI excludes 0; dagger = old p."""
    for i, ep in enumerate(EP):
        g = P[(P.epoch == ep)].dropna(subset=[col])
        cp.session_dots(ax, xpos[ep], g[col], g.animal, EPOCH_LINE[ep], jitter=jitter, seed=seed + i)
        iv = kit.boot_ci({a: v[col].tolist() for a, v in g.groupby("animal")}, rng) if len(g) else None
        x = xpos[ep] + dx
        row = i * 2 if rows_by_epoch else 0
        star = bool(iv is not None and (iv.lo > 0 or iv.hi < 0))
        if iv is not None:
            ax.errorbar(x, iv.point, yerr=[[iv.point - iv.lo], [iv.hi - iv.point]], fmt="o", color=EPOCH_LINE[ep],
                        ms=6, lw=1.8, capsize=0, mec="w", mew=0.6, zorder=5)
        if star:
            sig_marks.star_top(ax, x, "*", color=EPOCH_LINE[ep], row=row, fontsize=11 if rows_by_epoch else 12,
                               step=0.045 if rows_by_epoch else sig_marks.STEP)
        p = float(old_p.get(ep, np.nan))
        if np.isfinite(p) and p < 0.05:
            sig_marks.star_top(ax, x, DAGGER, color=EPOCH_LINE[ep], row=row + 1, fontsize=9,
                               step=0.045 if rows_by_epoch else sig_marks.STEP)
        per = g.groupby("animal")[col].mean()
        pt = iv.point if iv is not None else np.nan
        same = int(((per > 0) if pt > 0 else (per < 0)).sum()) if np.isfinite(pt) else 0
        RECORDS.append(dict(panel=panel, epoch=ep, scale="level (vs 0)", point=pt,
                            lo=iv.lo if iv is not None else np.nan, hi=iv.hi if iv is not None else np.nan,
                            k_same=same, n_animals=len(per),
                            per_animal="; ".join(f"{a} {d:+.3g}" for a, d in per.items()),
                            new_star="*" if star else "", lmm_q=p,
                            old_star=("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
                            if np.isfinite(p) else ""))
    ax.axhline(0, color="0.6", lw=0.8, zorder=0)


FAMILIES = {"pooled": fig_pooled, "fr": fig_fr, "fr_detail": lambda r: fig_fr(r, detail=True), "hemo": fig_hemo,
            "stillness": fig_stillness, "t12": fig_t12, "decay": fig_decay, "t13": fig_t13}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(FAMILIES))
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for k in a.only:
        RECORDS_START = len(RECORDS)
        FAMILIES[k](np.random.default_rng(0))     # one seeded rng per family: a family's draws do not depend on order
        print(k, len(RECORDS) - RECORDS_START, "cells")
    R = pd.DataFrame(RECORDS)
    R["changed"] = (R.new_star != "") != (R.old_star != "")
    tag = "_".join(a.only) if set(a.only) != set(FAMILIES) else "all"
    R.to_csv(OUT / f"star_changes_{tag}.csv", index=False)
    print(R[R.changed][["panel", "epoch", "old_star", "lmm_q", "new_star", "point", "lo", "hi", "k_same",
                        "n_animals"]].round(4).to_string(index=False))
    print(f"{int(R.changed.sum())} of {len(R)} cells change (star present / absent)")


if __name__ == "__main__":
    main()

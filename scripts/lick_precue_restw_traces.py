"""CUE-ALIGNED TRACES against a rest reference (Priya 10-28: "plot pre-cue traces" normalized to restw) -- the time course
behind the pre-cue-level maps of `lick_precue_restw.py`.

DEFAULT SINCE 11-09 (Priya approved 11-09): THE TRIAL SET OF THE LICK-FREE ENL LEVEL, RESTnl PRIMARY + SESSION-MEAN TWIN.
  trials  exactly the trials of `lick_precue_restw`'s ENL cache (<OUT_W>/precue_enl_<src><W>s/<label>.npz): camera
          WORKING trials (contact + incomplete, working period, stopped runs and quit tail removed) that HAVE a lick-free
          ENL window (`enl_window.sliding_lags`, W = `lp_licks.enl_post_s()`: 1.0 s under LP licks); the others dropped.
  value   mean over those trials of the cue-aligned area trace (-2..+3 s, construction `lick_precue_restw.area_series`)
          minus the reference: --rest-variant restnl (DEFAULT, RESTnl rule wr), session (NO subtraction: the dF/F is
          session-mean referenced), restw (old RESTW, robustness). The per-trial ENL window is not a fixed band on
          this time axis: the gray band marks the UNSLID window [-W, 0); the caption gives the share of slid trials.
  sanity  per session the mean over trials of each trial's ENL-window mean of these traces equals the level script's
          cache value (window_mean of the same samples; max |diff| printed) and its sessions csv.
  cache   <OUT_W>/precue_enl_traces_<src><W>s/<label>.npz (raw traces + the reference levels: one entry serves all three
          references); outputs precue_enl_traces_*<sfx>.csv, figures/lick_cued_precue_enl_traces[_farcontra][_change]
          <sfx>.png, sfx = _restnl | _sessmean | _restw. Run lick_precue_restw.py first (same env).
THE FIXED-WINDOW VERSION BELOW (engaged trials, restw / restnl) stays reachable with --precue-window fixed.

DEFINITION (per session x area x lag, -2..+3 s around the cue on the 0.05 s grid LAGS):
  trace  = exactly the construction of `lick_precue_restw.session_worker` / `lick_cued_ramp.wide_session` (ROI v8 masks,
           area means of U_atlas on the adopted SVT, slow drift basis removed, x1e3, NO per-trial baseline), with the
           ALM / tjM1 / MOs_rest / MOp_rest split (`iti_licks.with_rest`, whole-session series).
  value  = mean over ENGAGED trials (engaged & finite cue; the precue trial set, NOT only the first-lick trials stored in
           cued_wide) of trace(t) minus the session's RESTW baseline, read from the <OUT_W>/precue_restw/<label>.npz cache
           (the same numbers as the maps; restw_ok False sessions left out). Two trial sets: all spouts (>= 10 trials) and
           far-contra target only (>= 5 trials), as precue_restw.
  aggregation: animal x epoch mean over sessions -> mean of animals; shade = SEM over animals; solid where all 4 animals
           contribute, dotted where fewer.  CHANGE figure: each animal's epoch curve minus its own pre curve; per-bin
           significance = `iti_licks.sig_compute` (paired nested bootstrap animals > sessions per bin, BH-FDR q = 0.05 over
           the bins of the curve), drawn with `sig_draw`.
  first-lick tick: per epoch the median over sessions of the session's median first-lick latency (first lick onset in
           cue..stop, the `_first_lick_times` rule) on the plotted trial set.
Sanity: the -1..0 s mean of these traces is compared with precue_restw_sessions[_farcontra].csv (max |diff| printed).
Outputs (under iti_licks.OUT_W): precue_restw_traces/<label>.npz, precue_restw_traces_levels_per_animal[_farcontra].csv,
figures/lick_cued_precue_restw_traces[_farcontra][_change].png, figdata/<stem>.pkl. AUGUST-only pre (il.labels()).
RESTnl VARIANT (`--rest-variant restnl`, default restw unchanged): the same traces minus the LICK-FREE rest level and on
the CURRENT engaged gate, both read from <OUT_W>/precue_restnl/<label>.npz (lick_precue_restw.py --rest-variant restnl);
cache precue_restnl_traces/, every output file name with the suffix _restnl.
Usage: python scripts/lick_precue_restw_traces.py [--jobs N] [--plot-only] [--rest-variant restnl|session|restw]
       python scripts/lick_precue_restw_traces.py --precue-window fixed [--rest-variant restnl]     (the old version)
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import restnl  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

from wfield_local import mask_keys  # noqa: E402

CACHE = "precue_restw_traces"
SRC = "precue_restw"
VERSION = 1
LAGS = np.round(np.arange(-2.0, 3.0001, 0.05), 3)
WIN = (-1.0, 0.0)
AREAS = ["ALM", "tjM1", "SSp-m", "SSp-n", "MOs_rest", "MOp_rest", "SSp-bfd", "VISp"]
REF = {"MOs_rest", "MOp_rest", "SSp-bfd", "VISp"}
HEMI = (("right", "contralesional (right)"), ("left", "ipsilesional (left)"))
SETS = ((None, 10, ""), ("far_contra", 5, "_farcontra"))
STEM = "lick_cued_precue_restw_traces"
N_AN = 4


def session_worker(label: str) -> dict:
    """One session (module-level for spawn) -> <OUT_W>/precue_restw_traces/<label>.npz."""
    return _session(label, CACHE, SRC)


def session_worker_restnl(label: str) -> dict:
    """RESTnl variant -> <OUT_W>/precue_restnl_traces/<label>.npz (baseline + trial set of precue_restnl)."""
    return _session(label, CACHE_NL, SRC_NL)


CACHE_NL, SRC_NL = restnl.cache("precue_restnl_traces"), restnl.cache("precue_restnl")   # per RESTnl rule


def _cache(var: str) -> str:
    return CACHE if var == "restw" else CACHE_NL


def _session(label: str, cache: str, srcd: str) -> dict:
    f = il.OUT_W / cache / f"{label}.npz"
    if f.exists():
        z = np.load(f)
        if int(z["version"]) == VERSION:
            mask_keys.check_obj(z, f)             # stale-read guard (11-09)
            return {"label": label, "cached": True}
    from wfield_local import combined_encoding as ce
    src = np.load(il.OUT_W / srcd / f"{label}.npz")
    mask_keys.check_obj(src, il.OUT_W / srcd / f"{label}.npz")   # stale-read guard (11-09)
    d = il._load(label)
    # == lick_precue_restw.session_worker's trace construction
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    if cols != list(d["cols"]):
        raise ValueError(f"{label}: ROI columns differ from the session file")
    Ra = np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    Rs, cs = il.with_rest(R, cols, label, basis="split")
    if cs != [str(c) for c in src["cols"]]:
        raise ValueError(f"{label}: columns differ from the precue_restw cache")
    B = il._behaviour(label)
    T = B.trials
    Eg = T[T.engaged.astype(bool) & np.isfinite(T.cue_s.to_numpy(float))]
    cue = Eg.cue_s.to_numpy(float)
    pos = np.asarray(Eg.pos.astype(str).to_numpy(), str)
    if not np.array_equal(pos, src["pos"].astype(str)):
        raise ValueError(f"{label}: engaged trials differ from the {srcd} cache")
    X = il._sample(Rs, ft, cue, LAGS).astype(np.float64)
    out = (cue[:, None] + LAGS[None, :] < ft[0]) | (cue[:, None] + LAGS[None, :] > ft[-1])
    X[out] = np.nan
    # first-lick latency per engaged trial (rule of il._first_lick_times; NaN = no lick in cue..stop)
    bo = np.sort(np.asarray(B.lick_on_s, float)[np.isfinite(B.lick_on_s)])
    rt = np.full(len(cue), np.nan)
    for i, (c, s) in enumerate(zip(cue, Eg.stop_s.to_numpy(float))):
        s = s if np.isfinite(s) else c + 3.5
        j = np.searchsorted(bo, c)
        if j < len(bo) and bo[j] <= s:
            rt[i] = bo[j] - c
    sets = {}
    for sp, mn, tag in SETS:
        m = np.ones(len(pos), bool) if sp is None else pos == sp
        n = int(m.sum())
        sets[tag] = (np.nanmean(X[m], 0) - src["restw"][None, :]) if n >= mn else np.full((len(LAGS), len(cs)), np.nan)
        sets[tag + "_n"] = n
        sets[tag + "_rt"] = float(np.nanmedian(rt[m])) if np.isfinite(rt[m]).any() else np.nan
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, version=VERSION, cols=np.array(cs), lags=LAGS, restw_ok=bool(src["restw_ok"]),
                        animal=str(src["animal"]), epoch=str(src["epoch"]), all=sets[""], far=sets["_farcontra"],
                        n_all=sets["_n"], n_far=sets["_farcontra_n"], rt_all=sets["_rt"], rt_far=sets["_farcontra_rt"],
                        **mask_keys.stamp())
    return {"label": label, "cached": False}


def long_table(labels: list, key: str, var: str = "restw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Session x area x lag rows (restw_ok sessions, plotted areas only) + per-session first-lick medians."""
    rows, rts = [], []
    for x in labels:
        z = np.load(il.OUT_W / _cache(var) / f"{x}.npz")
        mask_keys.check_obj(z, il.OUT_W / _cache(var) / f"{x}.npz")   # stale-read guard (11-09)
        if not bool(z["restw_ok"]):
            continue
        A = z[key]
        if not np.isfinite(A).any():
            continue
        cs = [str(c) for c in z["cols"]]
        an, ep = str(z["animal"]), str(z["epoch"])
        rts.append({"label": x, "animal": an, "epoch": ep, "rt": float(z[f"rt_{key}"])})
        for a in AREAS:
            for h, _ in HEMI:
                c = f"{a}_{h}"
                if c not in cs:
                    continue
                rows.append(pd.DataFrame({"label": x, "animal": an, "epoch": ep, "area": a, "hemi": h, "lag": LAGS,
                                          "v": A[:, cs.index(c)]}))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(rts)


def curves(t: pd.DataFrame, change: bool) -> dict:
    """{epoch: (mean, sem, n_animals) over lags} from the session rows of one area x hemisphere."""
    P = t.groupby(["animal", "epoch", "lag"]).v.mean().unstack("lag").reindex(columns=LAGS)
    out = {}
    for ep in il.EP4:
        if ep not in P.index.get_level_values("epoch"):
            continue
        E = P.xs(ep, level="epoch")
        if change:
            if "pre" not in P.index.get_level_values("epoch"):
                continue
            pre = P.xs("pre", level="epoch")
            E = E.sub(pre.reindex(E.index))
        Y = E.to_numpy(float)
        n = np.isfinite(Y).sum(0)
        mu = np.nanmean(Y, 0)
        sd = np.nanstd(Y, 0, ddof=1) if Y.shape[0] > 1 else np.full(Y.shape[1], np.nan)
        out[ep] = (mu, sd / np.sqrt(np.maximum(n, 1)), n)
    return out


def sanity(labels: list, var: str = "restw") -> None:
    sfx = restnl.SFX if var == "restnl" else ""
    for sp, _mn, tag in SETS:
        key = "all" if sp is None else "far"
        L, _ = long_table(labels, key, var)
        w = (L.lag >= WIN[0] - 1e-9) & (L.lag <= WIN[1] + 1e-9)
        G = L[w].groupby(["label", "area", "hemi"]).v.mean().reset_index()
        G["area"] = G.area + "_" + G.hemi
        C = pd.read_csv(il.OUT_W / f"precue_restw_sessions{tag}{sfx}.csv")
        J = C.merge(G[["label", "area", "v"]], on=["label", "area"])
        d = (J.v - J.precue).abs()
        print(f"== sanity{tag or ' (all spouts)'}: -1..0 s mean of the traces vs precue_restw_sessions{tag}.csv: "
              f"max |diff| = {d.max():.2e} over {len(J)} session x area rows ({J.label.nunique()} sessions; csv rows "
              f"for plotted areas {C[C.area.isin(G.area.unique())].shape[0]})")


def compute(labels: list, var: str = "restw", enl: str | None = None) -> None:
    """Tables + figdata + figures; ``enl`` = the ENL reference (restnl / session / restw) of the ENL path, None = the
    fixed-window path (``var``)."""
    import lick_precue_restw as pr
    sfx = pr.ref_sfx(enl) if enl else (restnl.SFX if var == "restnl" else "")
    prefix, stem0 = ("precue_enl_traces", ENL_STEM) if enl else ("precue_restw_traces", STEM)
    for sp, _mn, tag in SETS:
        key = "all" if sp is None else "far"
        L, RT = long_table_enl(labels, key, enl) if enl else long_table(labels, key, var)
        # rule 8 table: per animal x epoch level at -2..-1 and -0.5..0 s (contra + ipsi key areas)
        rec = []
        for (a, h), g in L.groupby(["area", "hemi"]):
            for nm, (t0, t1) in (("lvl_m2_m1", (-2.0, -1.0)), ("lvl_m05_0", (-0.5, 0.0)), ("lvl_m1_0", WIN)):
                s = g[(g.lag >= t0 - 1e-9) & (g.lag <= t1 + 1e-9)].groupby(["animal", "epoch", "label"]).v.mean()
                for (an, ep), v in s.groupby(["animal", "epoch"]).mean().items():
                    rec.append({"area": f"{a}_{h}", "animal": an, "epoch": ep, "window": nm, "v": v})
        PA = pd.DataFrame(rec).pivot_table(index=["area", "animal"], columns=["window", "epoch"], values="v")
        PA.to_csv(il.OUT_W / f"{prefix}_levels_per_animal{tag}{sfx}.csv")
        ks = [f"{a}_right" for a in AREAS[:4]]
        sub = PA.loc[PA.index.get_level_values("area").isin(ks)]
        cols = [(w, e) for w in ("lvl_m2_m1", "lvl_m05_0") for e in ("pre", "acute") if (w, e) in sub.columns]
        print(f"== per animal (rule 8){tag}{sfx}: level vs {enl or ('RESTnl' if sfx else 'restw')} (dF/F x1e3), "
              "contralesional "
              "key areas\n"
              + sub[cols].round(2).to_string())
        rt = RT.groupby(["animal", "epoch"]).rt.median().groupby("epoch").median().to_dict()   # per-epoch tick
        rt_pa = RT.groupby(["epoch", "animal"]).rt.median().unstack().round(3)
        print(f"== median first-lick latency (s){tag}: per animal\n{rt_pa.to_string()}")
        rt = RT.groupby("epoch").rt.median().to_dict()
        for change in (False, True):
            pan, sig, recs = {}, {}, []
            for (a, h), g in L.groupby(["area", "hemi"], sort=False):
                pan[(a, h)] = curves(g, change)
                if change:
                    S = il.sig_compute(g.rename(columns={"v": "val"}), value="val", tag={"area": a, "hemi": h})
                    sig[(a, h)] = S
                    recs += S["recs"]
            stem = stem0 + tag + ("_change" if change else "") + sfx
            if change:
                pd.DataFrame(recs).to_csv(il.OUT_W / f"{prefix}_sigbins{tag}{sfx}.csv", index=False)
            n_sess = L.groupby("epoch").label.nunique().to_dict()
            n_an = L.groupby("epoch").animal.nunique().to_dict()
            extra = {}
            if enl:
                extra = {"enl": enl, "win_s": pr.enl_win_s(),
                         "slid_pct": 100.0 * RT.n_slid.sum() / max(int(RT.n.sum()), 1)}
            save_fd(stem, {"pan": pan, "sig": sig, "rt": rt, "change": change, "tag": tag, "n_sess": n_sess,
                           "n_an": n_an, "sfx": sfx, **extra})
            draw(stem)


def trial_set_sentence(spout: str | None = None, var: str | None = None, window: str | None = None) -> str:
    """THE trial + frame set of a figure (Priya 11-09): `lick_precue_restw.trial_set_sentence` (the same trials, read
    from / checked equal to its cache per session, and the same reference)."""
    import lick_precue_restw as pr
    return pr.trial_set_sentence(spout, var, window)


def draw(stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import EPOCH_LINE

    from wfield_local import trial_set_text as tst
    F = load_fd(stem)
    enl = F.get("enl")                                          # the ENL reference (11-09), None = fixed window
    if enl:
        import lick_precue_restw as pr
        tst.activate(None, text=pr.enl_trial_set_sentence("far_contra" if F.get("tag") else None, enl))
    else:
        tst.activate(None, text=trial_set_sentence("far_contra" if F.get("tag") else None,
                                                   "restnl" if F.get("sfx") else "restw", "fixed"))  # (11-09)
    ch = F["change"]
    rl = restnl.relabel if (F.get("sfx") and not enl) else (lambda t: t)
    win = (-float(F["win_s"]), 0.0) if enl else WIN
    fig, axs = plt.subplots(len(AREAS), 2, figsize=(11, 2.3 * len(AREAS)), sharex=True, squeeze=False,
                            layout="constrained")
    for i, a in enumerate(AREAS):
        for j, (h, hl) in enumerate(HEMI):
            ax = axs[i, j]
            ax.axvspan(*win, color="0.88", zorder=0)
            ax.axvline(0, color="0.3", lw=0.8)
            ax.axhline(0, color="0.5", lw=0.6, ls="-")
            if (a, h) not in F["pan"]:
                ax.text(0.5, 0.5, "no ROI in this hemisphere", transform=ax.transAxes, ha="center", fontsize=8)
                ax.set_yticks([])
                ax.set_title(f"{a} {hl}", fontsize=9)
                ax.tick_params(labelsize=7)
                continue
            for ep, (mu, se, n) in F["pan"][(a, h)].items():
                c = EPOCH_LINE[ep]
                ax.fill_between(LAGS, mu - se, mu + se, color=c, alpha=0.15, lw=0)
                ax.plot(LAGS, mu, color=c, lw=1.1, ls=":")
                ax.plot(LAGS, np.where(n >= N_AN, mu, np.nan), color=c, lw=1.4, label=ep)
            if ch:
                il.sig_draw(ax, F["sig"][(a, h)], EPOCH_LINE)
                il._headroom([ax], 0.15)
            for ep, r in F["rt"].items():
                if np.isfinite(r) and (not ch or ep != "pre"):
                    ax.plot([r, r], [0.0, 0.07], color=EPOCH_LINE[ep], lw=2.2, transform=ax.get_xaxis_transform())
            ax.set_title(f"{a} {hl}" + ("  [reference]" if a in REF else ""), fontsize=9)
            ax.set_xlim(LAGS[0], LAGS[-1])
            if j == 0:
                ax.set_ylabel("change from pre\n(dF/F x1e3)" if ch else
                              (pr.ref_unit(enl).replace(" (", "\n(") if enl else rl("dF/F x1e3\nabove restw")),
                              fontsize=8)
            ax.tick_params(labelsize=7)
    for ax in axs[-1]:
        ax.set_xlabel("time from cue (s)", fontsize=8)
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside lower center", fontsize=9, frameon=False, ncol=4)
    what = "far-contra target trials only" if F["tag"] else "all spouts"
    ns = ", ".join(f"{e} {F['n_an'].get(e, 0)} animals / {F['n_sess'].get(e, 0)} sessions" for e in il.EP4)
    body = ("CHANGE from each animal's own pre curve (epoch minus pre per animal, mean over animals, shade = SEM over "
            "animals); bars at the top = bins where the epoch differs from pre (paired nested bootstrap animals > "
            "sessions per bin, BH-FDR q = 0.05 over the bins of each curve; iti_licks.sig_compute)"
            if ch else "level above the session's RESTW baseline (position-weighted flat rest median, restdock05, "
            "engagement-gated); per animal x epoch mean over sessions, then mean over animals, shade = SEM over animals")
    import textwrap
    if enl:
        lvl = ("" if ch else f"{pr.ref_text(enl)}; per animal x epoch mean over sessions, then mean over animals, shade "
               "= SEM over animals")
        body = (body if ch else lvl) + (f". Trials: camera WORKING trials with a {pr.enl_window_text()} "
                                        f"({F.get('slid_pct', float('nan')):.0f}% of kept trials slid earlier); gray = "
                                        "the unslid ENL window")
        fig.suptitle(textwrap.fill(f"NEURAL: cue-aligned area traces, {what} -- {body}. Solid = all 4 animals "
                                   f"contribute, dotted = fewer. Ticks at the bottom = median first-lick latency per "
                                   f"epoch. Rows = areas (ALM / tjM1 split from MOs / MOp, ROI v8; last four = "
                                   f"references; VISp has a contralesional ROI only). n: {ns}.", 150), fontsize=9)
        (il.OUT_W / "figures").mkdir(parents=True, exist_ok=True)
        save_png_svg(fig, il.OUT_W / "figures" / f"{stem}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return
    body = rl(body) + (" Trials = CURRENT engaged gate" if F.get("sfx") else "")
    fig.suptitle(textwrap.fill(rl("NEURAL: cue-aligned area traces minus RESTW") + f", ENGAGED trials, {what} -- {body}. Solid = all 4 "
                 f"animals contribute, dotted = fewer. Gray = -1..0 s window of the pre-cue maps; ticks at the bottom = "
                 f"median first-lick latency per epoch. Rows = areas (ALM / tjM1 split from MOs / MOp, ROI v8; last four "
                 f"= references; VISp has a contralesional ROI only). n: {ns}.", 150), fontsize=9)
    (il.OUT_W / "figures").mkdir(parents=True, exist_ok=True)
    save_png_svg(fig, il.OUT_W / "figures" / f"{stem}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main_fixed() -> int:
    """The FIXED-window version (--precue-window fixed): unchanged since 11-09 apart from this entry point."""
    from wfield_local import analysis_kit as ak
    var = restnl.variant()
    sfx = restnl.SFX if var == "restnl" else ""
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence(None, var, "fixed"))    # draw() narrows it per figure
    if plot_only():
        for _sp, _mn, tag in SETS:
            for c in ("", "_change"):
                draw(STEM + tag + c + sfx)
        return 0
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8
    labs = il.labels()
    worker = session_worker_restnl if var == "restnl" else session_worker
    _res, fails = ak.fan_sessions(labs, worker, jobs=min(jobs, 4 if var == "restnl" else 8))
    if fails:
        raise SystemExit(f"{len(fails)} sessions failed: {fails}")
    sanity(labs, var)
    compute(labs, var)
    return 0


# ==================================================================================================================
# THE LICK-FREE ENL TRIAL SET (default since 11-09, Priya approved): see the module docstring, first block.
# ==================================================================================================================
ENL_CACHE_T = "precue_enl_traces"
ENL_VERSION_T = 1
ENL_STEM = "lick_cued_precue_enl_traces"
#: the reference levels copied from the level cache, so one traces entry serves every reference
_REF_FIELDS = ("restnl_wr", "restnl_wr_ok", "restnl_strict", "restnl_strict_ok", "restw", "restw_ok")


def enl_cache_t() -> str:
    """precue_enl_traces_<src><W>s: the traces twin of `lick_precue_restw.enl_cache()`."""
    import lick_precue_restw as pr
    return ENL_CACHE_T + pr.enl_cache()[len(pr.ENL_CACHE):]


def session_worker_enl(label: str) -> dict:
    """One session (module-level for spawn) -> <OUT_W>/<enl_cache_t()>/<label>.npz, from the level script's ENL cache
    (its trials, windows and reference levels)."""
    import lick_precue_restw as pr

    from wfield_local import enl_window
    src = np.load(il.OUT_W / pr.enl_cache() / f"{label}.npz")
    f = il.OUT_W / enl_cache_t() / f"{label}.npz"
    if f.exists():
        z = np.load(f)
        if (int(z["version"]) == ENL_VERSION_T and str(z["meta"]) == str(src["meta"])
                and np.array_equal(z["cue"], src["cue"])):
            return {"label": label, "cached": True}
    Rs, cs, ft, _d = pr.area_series(label)
    if cs != [str(c) for c in src["cols"]]:
        raise ValueError(f"{label}: columns differ from the {pr.enl_cache()} cache")
    cue, lo, hi = src["cue"].astype(float), src["lo"].astype(float), src["hi"].astype(float)
    pos, stop = src["pos"].astype(str), src["stop"].astype(float)
    X = pr.sample_cues(Rs, ft, cue)
    lv = enl_window.window_mean(X, LAGS, lo, hi)                # each trial's ENL-window level from these samples
    maxdiff = float(np.nanmax(np.abs(lv - src["p10"]))) if len(cue) else np.nan
    B = il._behaviour(label)                                    # first-lick latency (rule of il._first_lick_times)
    bo = np.sort(np.asarray(B.lick_on_s, float)[np.isfinite(B.lick_on_s)])
    rt = np.full(len(cue), np.nan)
    for i, (c, s) in enumerate(zip(cue, stop)):
        s = s if np.isfinite(s) else c + 3.5
        j = np.searchsorted(bo, c)
        if j < len(bo) and bo[j] <= s:
            rt[i] = bo[j] - c
    slid = np.abs(lo + float(pr.enl_win_s())) > 1e-3
    sets = {}
    for sp, mn, tag in SETS:
        m = np.ones(len(pos), bool) if sp is None else pos == sp
        n = int(m.sum())
        ok = n >= mn
        sets[tag] = np.nanmean(X[m], 0) if ok else np.full((len(LAGS), len(cs)), np.nan)
        sets[tag + "_lvl"] = np.nanmean(lv[m], 0) if ok else np.full(len(cs), np.nan)
        sets[tag + "_n"] = n
        sets[tag + "_slid"] = int((slid & m).sum())
        sets[tag + "_rt"] = float(np.nanmedian(rt[m])) if np.isfinite(rt[m]).any() else np.nan
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, version=ENL_VERSION_T, meta=src["meta"], cue=cue, cols=np.array(cs), lags=LAGS,
                        animal=str(src["animal"]), epoch=str(src["epoch"]), all=sets[""], far=sets["_farcontra"],
                        lvl_all=sets["_lvl"], lvl_far=sets["_farcontra_lvl"], n_all=sets["_n"],
                        n_far=sets["_farcontra_n"], slid_all=sets["_slid"], slid_far=sets["_farcontra_slid"],
                        rt_all=sets["_rt"], rt_far=sets["_farcontra_rt"], maxdiff=maxdiff,
                        **{k: src[k] for k in _REF_FIELDS})
    return {"label": label, "cached": False, "maxdiff": maxdiff}


def long_table_enl(labels: list, key: str, ref: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`long_table` on the ENL traces: session x area x lag rows of (trace - reference) for sessions with a valid
    reference and >= the set's trials, + per-session first-lick medians and kept / slid counts."""
    import lick_precue_restw as pr
    rows, rts = [], []
    for x in labels:
        z = np.load(il.OUT_W / enl_cache_t() / f"{x}.npz")
        base, ok = pr.ref_base(z, ref)
        A = z[key]
        if not ok or not np.isfinite(A).any():
            continue
        A = A - base[None, :]
        cs = [str(c) for c in z["cols"]]
        an, ep = str(z["animal"]), str(z["epoch"])
        rts.append({"label": x, "animal": an, "epoch": ep, "rt": float(z[f"rt_{key}"]), "n": int(z[f"n_{key}"]),
                    "n_slid": int(z[f"slid_{key}"])})
        for a in AREAS:
            for h, _ in HEMI:
                c = f"{a}_{h}"
                if c not in cs:
                    continue
                rows.append(pd.DataFrame({"label": x, "animal": an, "epoch": ep, "area": a, "hemi": h, "lag": LAGS,
                                          "v": A[:, cs.index(c)]}))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(rts)


def sanity_enl(labels: list, ref: str) -> None:
    """Per session x area: the mean over trials of each trial's ENL-window level (these traces) minus the reference
    vs the level script's sessions csv; and the per-trial window means vs its cache."""
    import lick_precue_restw as pr
    sfx = pr.ref_sfx(ref)
    md = [float(np.load(il.OUT_W / enl_cache_t() / f"{x}.npz")["maxdiff"]) for x in labels]
    print(f"== sanity: per-trial ENL-window level from these traces vs the {pr.enl_cache()} cache: max |diff| = "
          f"{np.nanmax(md):.2e} over {len(md)} sessions")
    for sp, _mn, tag in SETS:
        key = "all" if sp is None else "far"
        rec = []
        for x in labels:
            z = np.load(il.OUT_W / enl_cache_t() / f"{x}.npz")
            base, _ok = pr.ref_base(z, ref)
            for c, v in zip([str(c) for c in z["cols"]], z[f"lvl_{key}"] - base):
                rec.append({"label": x, "area": c, "v": v})
        C = pd.read_csv(il.OUT_W / f"{pr.ENL_PREFIX}_sessions{tag}{sfx}.csv")
        J = C.merge(pd.DataFrame(rec), on=["label", "area"])
        print(f"== sanity{tag or ' (all spouts)'}: trace ENL levels vs {pr.ENL_PREFIX}_sessions{tag}{sfx}.csv: max |diff| "
              f"= {(J.v - J.precue).abs().max():.2e} over {len(J)} session x area rows ({J.label.nunique()} sessions)")


def main() -> int:
    import lick_precue_restw as pr
    if pr.window_mode() == "fixed":
        return main_fixed()
    from wfield_local import analysis_kit as ak
    from wfield_local import trial_set_text as tst
    ref = pr.reference("enl")
    sfx = pr.ref_sfx(ref)
    tst.declare(text=pr.enl_trial_set_sentence(None, ref))      # draw() narrows it per figure
    if plot_only():
        for _sp, _mn, tag in SETS:
            for c in ("", "_change"):
                draw(ENL_STEM + tag + c + sfx)
        return 0
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    labs = il.labels()
    _res, fails = ak.fan_sessions(labs, session_worker_enl, jobs=min(jobs, 6))
    if fails:
        raise SystemExit(f"{len(fails)} sessions failed: {fails}")
    sanity_enl(labs, ref)
    compute(labs, enl=ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

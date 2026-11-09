"""PRE-CUE LEVEL per area against the RESTW baseline -- the extra row for the cued area maps (Priya 10-28, approved).

WHY A REST REFERENCE. The spout is in place ~2-3 s before the cue, so every in-trial window (incl. the -1.0..-0.5 s cue
baseline of the cued maps) already contains the pre-cue state; an in-trial baseline cannot measure it. The pre-cue level
is therefore referenced to the session's REST baseline instead.

DEFINITION (per session x area):
  trace   = exactly the area traces of `lick_cued_ramp.wide_session` / `iti_licks.extract` (ROI v8 masks, area means of
            U_atlas on the adopted hemo-corrected SVT, slow drift basis removed, x1e3 -- NO per-trial baseline), with the
            ALM / tjM1 / MOs_rest / MOp_rest split of the cued maps (`iti_licks.with_rest`, applied to the whole-session
            time series BEFORE the baseline, since the median is not linear).
  restw   = `rest_by_position.restw_from_frames` on that whole-session area time series: per position the MEDIAN over
            its rest frames, positions averaged with equal weight (all 6 required, >= 200 frames each). Rest frames =
            `rest_frames_by_position(session, T, docked=False, engaged_only=True)` -- the same call as the encoder's
            `_restw_baseline` and `position_reference_maps.session_restw_svt` (both docked=False): the rest MASK is
            already the configured `restdock05` variant (docked spout, lick buffer [0.5, 1.0] s), so docked=False adds
            no second docked intersection; engagement-gated (quit-period rest excluded, docs/REST_ENGAGEMENT_AUDIT.md).
            A session that cannot form restw gets the unweighted rest median, FLAGGED (`restw_ok` False) and left OUT of
            every statistic and map.
  precue  = mean over ENGAGED trials (combined_encoding behavior table, engaged & finite cue) of the trial's mean trace
            over -1.0..0 s before the cue (0.05 s grid, as LAGS_W), minus restw. Secondary window -0.5..0 s (precue05).
            Also on the first-lick trials only (the cued maps' trial set; `precue_fl`) for the correlation check.
  per target spout: the same restricted to engaged trials at that target (>= 5 trials; all spouts >= 10).
Stats: change from pre per area / epoch with `analysis_kit.boot_delta` (nested animals > sessions), significance = CI
excludes 0, no multiple-comparison correction -- as `iti_area_maps.changes` (the cued maps). AUGUST-ONLY pre baseline
(session list = `iti_licks.labels()`, the August+ curated frozen-run set). Image LEFT = ipsilesional.
Outputs (under iti_licks.OUT_W): precue_restw/<label>.npz (per-session cache), precue_restw_sessions[_<spout>].csv,
precue_restw_changes[_<spout>].csv, precue_restw_session_notes.csv, figures/lick_cued_precue_restw_maps[_farcontra].png
(+ _levels.png), figdata/<stem>.pkl.
RESTnl VARIANT (Priya 10-29, `--rest-variant restnl`; default restw = everything above, unchanged): the baseline is the
LICK-FREE rest level of `scripts/restnl.py` (restw_strict_check strict_w50; sessions without it left out), the trial set
is the CURRENT engaged gate (the behavior table now; the precue_restw cache holds an older gate in 86 / 107 sessions).
Cache <OUT_W>/precue_restnl/<label>.npz (field `restw` = the RESTnl level there, `restw_old` = RESTW for comparison);
every output file name gets the suffix _restnl.
Usage: python scripts/lick_precue_restw.py [--jobs N] [--plot-only] [--rest-variant restnl]
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

CACHE = "precue_restw"
VERSION = 2
LAGS = np.round(np.arange(-2.0, 3.0001, 0.05), 3)          # = lick_cued_ramp.LAGS_W (validation against cued_wide)
WIN = (-1.0, 0.0)
WIN2 = (-0.5, 0.0)
VALS = (("precue", "PRE-CUE LEVEL -1.0..0 s above restw"), ("precue05", "pre-cue level -0.5..0 s above restw"))
KEY = ["ALM", "tjM1", "SSp-m", "SSp-n"]


def session_worker(label: str) -> dict:
    """One session (module-level for spawn) -> <OUT_W>/precue_restw/<label>.npz; returns the notes."""
    return _session(label, "restw")


def session_worker_restnl(label: str) -> dict:
    """RESTnl variant (module-level for spawn) -> <OUT_W>/precue_restnl/<label>.npz (current engaged gate)."""
    return _session(label, "restnl")


def _cache(var: str) -> str:
    return CACHE if var == "restw" else restnl.cache("precue_restnl")   # strict rule = precue_restnl, wr = _wr


def _session(label: str, var: str) -> dict:
    f = il.OUT_W / _cache(var) / f"{label}.npz"
    if f.exists():
        z = np.load(f)
        if int(z["version"]) == VERSION:
            mask_keys.check_obj(z, f)             # stale-read guard: built on the current analysis masks (11-09)
            return {"label": label, "cached": True}
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local.rest_by_position import rest_frames_by_position, restw_from_frames
    d = il._load(label)
    # == lick_cued_ramp.wide_session's trace construction (validated below against the stored cued_wide cue traces)
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
    Rs, cs = il.with_rest(R, cols, label, basis="split")    # (T, cols + 4 rest areas)
    extra = {}
    if var == "restnl":                                        # the lick-free level (restw_strict_check, strict_w50)
        base, ok = restnl.level(label, cs)
        old = np.load(il.OUT_W / CACHE / f"{label}.npz")
        mask_keys.check_obj(old, il.OUT_W / CACHE / f"{label}.npz")
        if [str(c) for c in old["cols"]] != cs:
            raise ValueError(f"{label}: columns differ from the precue_restw cache")
        used, n_rest = old["used"], int(old["n_rest"])
        extra = {"restw_old": np.asarray(old["restw"], float), "restw_old_ok": bool(old["restw_ok"]),
                 "n_eng_old": len(old["pos"])}
    else:
        # restw on the whole-session area series
        s = next(x for x in config.load_sessions() if x["label"] == label)
        per, info = rest_frames_by_position(s, Rs.shape[0], docked=False, engaged_only=True)
        if info.get("error"):
            raise RuntimeError(f"{label}: no rest mask ({info['error']})")
        base, used = restw_from_frames(Rs.T, per, label=label)
        ok = base is not None
        n_rest = int(sum(len(v) for v in per.values()))
        if not ok:                                             # loud fallback: unweighted rest median, flagged
            idx = np.concatenate([np.asarray(v) for v in per.values()]) if per else np.array([], int)
            idx = idx[idx < Rs.shape[0]]
            base = np.median(Rs[idx], 0)[:, None] if idx.size else np.full((Rs.shape[1], 1), np.nan)
        base = base[:, 0]
    # engaged trials
    B = il._behaviour(label)
    T = B.trials
    Eg = T[T.engaged.astype(bool) & np.isfinite(T.cue_s.to_numpy(float))]
    cue = Eg.cue_s.to_numpy(float)

    def win_means(t):
        X = il._sample(Rs, ft, t, LAGS).astype(np.float64)
        out = (t[:, None] + LAGS[None, :] < ft[0]) | (t[:, None] + LAGS[None, :] > ft[-1])
        X[out] = np.nan
        w1 = (LAGS >= WIN[0]) & (LAGS <= WIN[1])
        w2 = (LAGS >= WIN2[0]) & (LAGS <= WIN2[1])
        return np.nanmean(X[:, w1], 1), np.nanmean(X[:, w2], 1), X

    p10, p05, _ = win_means(cue)
    # the cued maps' trial set (first-lick trials) + validation against cued_wide
    cc, _fl = il._first_lick_times(label, d["first_rt"])
    cc = cc[np.isfinite(cc)]
    f10, f05, Xf = win_means(cc)
    maxdiff = np.nan
    wf = il.OUT / "cued_wide" / f"{label}.npz"
    if wf.exists():
        z = np.load(wf)
        ref = z["cue"].astype(float)
        if ref.shape[0] == Xf.shape[0]:
            maxdiff = float(np.nanmax(np.abs(Xf[..., :len(cols)] - ref)))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, version=VERSION, cols=np.array(cs), restw=base, restw_ok=ok, used=np.array(used, int),
                        n_rest=n_rest, p10=p10, p05=p05, pos=np.asarray(Eg.pos.astype(str).to_numpy(), str), f10=f10, f05=f05,
                        maxdiff=maxdiff, animal=str(d["animal"]), epoch=str(d["epoch"]), **extra, **mask_keys.stamp())
    return {"label": label, "cached": False, "restw_ok": ok, "n_pos": len(used), "maxdiff": maxdiff}


def session_tables(labels: list, var: str = "restw") -> tuple[dict, pd.DataFrame]:
    """{spout or None: per-session x area table}, notes."""
    out, notes = {None: [], "far_contra": []}, []
    for x in labels:
        z = np.load(il.OUT_W / _cache(var) / f"{x}.npz")
        mask_keys.check_obj(z, il.OUT_W / _cache(var) / f"{x}.npz")
        cs = [str(c) for c in z["cols"]]
        ok = bool(z["restw_ok"])
        pos = z["pos"].astype(str)
        notes.append({"label": x, "animal": str(z["animal"]), "epoch": str(z["epoch"]), "restw_ok": ok,
                      "positions_used": " ".join(map(str, z["used"])), "n_rest_frames": int(z["n_rest"]),
                      "n_engaged": len(pos), "n_firstlick": len(z["f10"]), "maxdiff_vs_cued_wide": float(z["maxdiff"])})
        for sp, mn in ((None, 10), ("far_contra", 5)):
            m = np.ones(len(pos), bool) if sp is None else pos == sp
            if m.sum() < mn:
                continue
            a10 = np.nanmean(z["p10"][m], 0) - z["restw"]
            a05 = np.nanmean(z["p05"][m], 0) - z["restw"]
            fl = np.nanmean(z["f10"], 0) - z["restw"] if sp is None and len(z["f10"]) else np.full(len(cs), np.nan)
            for j, c in enumerate(cs):
                if c == "cortex" or c.rsplit("_", 1)[0] in ("MOs", "MOp"):
                    continue
                out[sp].append({"label": x, "animal": str(z["animal"]), "epoch": str(z["epoch"]), "area": c,
                                "precue": a10[j], "precue05": a05[j], "precue_fl": fl[j], "restw": z["restw"][j],
                                "restw_ok": ok, "n": int(m.sum())})
    return {k: pd.DataFrame(v) for k, v in out.items()}, pd.DataFrame(notes)


def per_animal(T: pd.DataFrame, val: str) -> pd.DataFrame:
    """Rule 8: area x animal, session-mean per epoch and change vs pre."""
    P = T.pivot_table(index=["area", "animal"], columns="epoch", values=val, aggfunc="mean").reindex(columns=il.EP4)
    for ep in il.EP4[1:]:
        P[f"d_{ep}"] = P[ep] - P["pre"]
    return P


def run(spout: str | None, T: pd.DataFrame, sfx: str = "") -> None:
    import iti_area_maps as iam
    from frozen_area_maps import outline
    tag = "" if spout is None else "_" + spout.replace("_", "")
    T = T[T.restw_ok]
    T.to_csv(il.OUT_W / f"precue_restw_sessions{tag}{sfx}.csv", index=False)
    pa = per_animal(T, "precue")
    pa.to_csv(il.OUT_W / f"precue_restw_per_animal{tag}{sfx}.csv")
    keyrows = pa[pa.index.get_level_values("area").str.rsplit("_", n=1).str[0].isin(KEY)]
    print(f"== per animal (rule 8), PRE-CUE LEVEL -1..0 s above {'RESTnl' if sfx else 'restw'}{tag}\n"
          + keyrows.round(2).to_string())
    C = pd.concat([iam.changes(T, v) for v, _t in VALS])
    C.to_csv(il.OUT_W / f"precue_restw_changes{tag}{sfx}.csv", index=False)
    print(C[C.sig].sort_values(["value", "epoch", "delta"]).round(3).to_string(index=False))
    labels = sorted(T.label.unique())
    img, nm = iam.split_image(labels[0])
    rows = []
    for val, ttl in VALS:
        c = C[C.value == val]
        lim = np.nanpercentile(np.abs(c.delta), 98) or 1.0
        pans = []
        for ep in il.EP4[1:]:
            ce_ = c[c.epoch == ep].set_index("area")
            pans.append((ep, ce_.delta.to_dict(), list(ce_.index[ce_.sig]), int(ce_.sig.sum())))
        rows.append((ttl, lim, pans))
    what = "CUED trials, all engaged trials" + ("" if spout is None else f", {spout.replace('_', '-')} target only")
    stem = f"lick_cued_precue_restw_maps{tag}{sfx}"
    save_fd(stem, {"img": img, "nm": nm, "ol": outline(img), "rows": rows, "what": what,
                   "levels": iam._levels(T, VALS), "sfx": sfx})
    draw(stem)


def draw(stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import iti_area_maps as iam
    import matplotlib.pyplot as plt
    from frozen_area_maps import outline, paint
    F = load_fd(stem)
    sfx = F.get("sfx", "")
    rl = restnl.relabel if sfx else (lambda t: t)
    base = stem[:-len(sfx)] if sfx else stem
    nm = F["nm"]
    img = iam.trim_rest(F["img"], nm)
    ol = outline(img)
    note = ("pre-cue level = mean dF/F (x1e3) over the window before the cue (spout already in place), engaged trials, "
            "minus the session's RESTW baseline (position-weighted flat rest median, restdock05, engagement-gated)")
    (il.OUT_W / "figures").mkdir(parents=True, exist_ok=True)
    nr = len(F["rows"])
    fig, axs = plt.subplots(nr, 3, figsize=(14, 4.5 * nr), squeeze=False)
    for i, (ttl, lim, pans) in enumerate(F["rows"]):
        for j, (ep, delta, sig_areas, n_sig) in enumerate(pans):
            ax = axs[i, j]
            im = ax.imshow(paint(img, nm, delta), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            iam._sig_halo(ax, img, nm, sig_areas, list(delta))
            ax.set_title(rl(f"{ttl}\n{ep} - pre ({n_sig} areas significant)"), fontsize=9)
            ax.axis("off")
        fig.colorbar(im, ax=axs[i, :], fraction=0.02, pad=0.01).set_label(rl("change in dF/F above restw (x1e3)"))
    fig.suptitle(rl(f"NEURAL, NO DECODER: {F['what']} -- change from pre per area in the PRE-CUE LEVEL; {note}; ALM / tjM1 "
                 "separate from the rest of MOs / MOp (ROI v8); black outline on a white halo = nested-bootstrap CI "
                 "excludes 0. Image left = ipsilesional." + (" Trials = CURRENT engaged gate." if sfx else "")),
                 fontsize=10, wrap=True)
    save_png_svg(fig, il.OUT_W / "figures" / f"{stem}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    nr = len(F["levels"])
    fig, axs = plt.subplots(nr, 4, figsize=(18, 4.5 * nr), squeeze=False)
    for i, (ttl, lim, pans) in enumerate(F["levels"]):
        for j, (ep, lev) in enumerate(pans):
            ax = axs[i, j]
            im = ax.imshow(paint(img, nm, lev), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            ax.set_title(rl(f"{ttl}\n{ep}"), fontsize=9)
            ax.axis("off")
        fig.colorbar(im, ax=axs[i, :], fraction=0.015, pad=0.01).set_label(rl("dF/F above restw (x1e3)"))
    fig.suptitle(rl(f"NEURAL, NO DECODER: {F['what']} -- per-epoch PRE-CUE LEVEL per area (mean over animals of each "
                 f"animal's session mean; not the change from pre); {note}; one color scale per row. Image left = "
                 "ipsilesional. Descriptive." + (" Trials = CURRENT engaged gate." if sfx else "")), fontsize=10, wrap=True)
    save_png_svg(fig, il.OUT_W / "figures" / f"{base}_levels{sfx}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def sanity(T: pd.DataFrame, notes: pd.DataFrame, sfx: str = "") -> None:
    print(f"== restw fallback sessions: {int((~notes.restw_ok).sum())}/{len(notes)} "
          f"{list(notes.label[~notes.restw_ok])}")
    print(f"== validation vs cued_wide cue traces: max |diff| = {notes.maxdiff_vs_cued_wide.max():.4f} "
          f"(NaN sessions {int(notes.maxdiff_vs_cued_wide.isna().sum())})")
    f = il.OUT / "area_auc_sessions_split_cued.csv"
    if f.exists() and not sfx:
        A = pd.read_csv(f)[["label", "area", "cue"]]
        J = T[T.restw_ok].merge(A, on=["label", "area"])
        r1 = np.corrcoef(J.precue, J.cue)[0, 1]
        r2 = np.corrcoef(J.precue_fl.astype(float), J.cue)[0, 1]
        rr = J.groupby("area").apply(lambda g: np.corrcoef(g.precue, g.cue)[0, 1], include_groups=False)
        print(f"== corr(pre-cue level, in-trial cue response AUC) over session x area: r = {r1:.3f} (first-lick trials "
              f"{r2:.3f}); within-area median r = {rr.median():.3f} [{rr.min():.3f}, {rr.max():.3f}], n = {len(J)}")
        print(f"== corr(pre-cue level all engaged vs first-lick trials): "
              f"{np.corrcoef(J.precue, J.precue_fl.astype(float))[0, 1]:.3f}")


def main() -> int:
    from wfield_local import analysis_kit as ak
    var = restnl.variant()
    sfx = restnl.SFX if var == "restnl" else ""
    if plot_only():
        for st in ("lick_cued_precue_restw_maps", "lick_cued_precue_restw_maps_farcontra"):
            draw(st + sfx)
        return 0
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8
    labs = il.labels()
    worker = session_worker_restnl if var == "restnl" else session_worker
    res, fails = ak.fan_sessions(labs, worker, jobs=min(jobs, 4 if var == "restnl" else 8))
    if fails:
        raise SystemExit(f"{len(fails)} sessions failed: {fails}")
    Ts, notes = session_tables(labs, var)
    notes.to_csv(il.OUT_W / f"precue_restw_session_notes{sfx}.csv", index=False)
    sanity(Ts[None], notes, sfx)
    run(None, Ts[None], sfx)
    run("far_contra", Ts["far_contra"], sfx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Quiet-event maps split by face movement x height x polarity x state, and per-event ipsi / contra timing (cohort1, 11-01).

Tests T1-T4 of docs/status/STATUS_2026-11-01_QUIET_EVENTS_HANDOFF.md (Priya 11-01: the acute orofacial event map in
Fig 15.7 "feels important"):
  T1  SELECTION: pre-stroke orofacial events are mostly face-movement-locked (whisking drives barrel / limb cortex) and
      acute ones are 5x more numerous, so the pre vs acute map difference may be WHICH events, not the events. Maps are
      split by face movement (FaceRhythm speed above the session's engaged-quiet 90th percentile within +/-0.3 s of the
      peak; quiet_spikes_v2_fr's `move_raw_*`) and by height (4-6 / 6-9 / > 9 x the animal's pre robust SD).
  T2  IPSI vs CONTRA: per event, peak amplitude of the left (ipsilesional) and right orofacial traces and the time each
      reaches half its peak (interpolated), on events detected in the bilateral orofacial trace (side-neutral).
  T3  UP vs DOWN: the same for negative events (sign-flipped trace).
  T4  STATE: events in the QUIT period (after the session's quit) kept separately from ENGAGED ones.
Events = quiet_spikes_v2_fr raw events (the v2 detector on the 2-s-median-detrended trace, quiet frames = no lick /
running within 1.5 s), height > 4 x the animal's August-pre robust SD. Maps = high-passed movie (quiet_fluct (2)),
stored 4x-downsampled; SPREAD index per class = mean over lateral SSp (SSp-bfd, SSp-ul, SSp-ll; both sides) / mean over
orofacial cortex (ALM, tjM1, SSp-m, SSp-n) at 0 and +0.2 s.

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_eta_split.py [--jobs 5] [--summarise]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import columns, session_traces  # noqa: E402
from quiet_spikes_v2 import V2, busy_frames, detrend  # noqa: E402

OUTS = V2 / "eta_split"
FRV = V2 / "fr_regressed_own" / "sessions"
K, MOVE_PCTL, DS = 4.0, 90.0, 4
HBINS = (4.0, 6.0, 9.0, np.inf)
MAP_LAGS_S = (-0.3, 0.0, 0.2, 0.5, 1.0)
TR_LAG_S = (-1.0, 2.0)
LAT_AREAS = ("SSp-bfd", "SSp-ul", "SSp-ll")
ORO_AREAS = ("ALM", "tjM1", "SSp-m", "SSp-n")


def classes():
    return [(kind, st, mv, h) for kind in ("pos", "neg") for st in ("eng", "quit") for mv in ("still", "move")
            for h in range(len(HBINS) - 1)]


def ckey(c):
    return f"{c[0]}_{c[1]}_{c[2]}_h{c[3]}"


def ds_img(v, flat, shape):
    img = np.full(int(np.prod(shape)), np.nan)
    img[flat] = v
    img = img.reshape(shape)
    H, W = (shape[0] // DS) * DS, (shape[1] // DS) * DS
    b = img[:H, :W].reshape(H // DS, DS, W // DS, DS)
    with np.errstate(invalid="ignore"):
        return np.nanmean(b, (1, 3)).astype(np.float32)


def half_time(y, p, lags_fr, sign):
    """Interpolated frame (relative to p) where sign*y first reaches half its peak on the rise, within -0.3 .. +0.15 s."""
    z = sign * y
    pk = z[p]
    lo = p + lags_fr[0]
    seg = z[lo:p + 1]
    half = pk / 2
    idx = np.flatnonzero(seg < half)
    if not len(idx) or pk <= 0:
        return np.nan
    j = idx[-1]
    if j + 1 >= len(seg):
        return np.nan
    return (lo + j + (half - seg[j]) / (seg[j + 1] - seg[j])) - p


def extract(label):
    f = OUTS / "sessions" / f"{label}.npz"
    if f.exists():
        return label
    fr_f = FRV / f"{label}.npz"
    if not fr_f.exists():
        return label                                                     # no face video -> no movement flag
    from iti_licks import roi_masks
    from quiet_fluct import animal_sd
    from scipy.ndimage import uniform_filter1d
    from wfield_local import combined_encoding as ce
    Xr, ft, fs, rest, engaged, _ = session_traces(label, detrend=False)
    V, ft2, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    T = Xr.shape[1]
    Xd = detrend(Xr, fs)
    cols = columns()
    sd = animal_sd()[label[:4]]
    z = np.load(fr_f)
    V = np.asarray(V, np.float64)
    Vh = V - uniform_filter1d(V, int(round(2.0 * fs)), axis=0, mode="nearest")
    rm = roi_masks(parea, pstat, names, flat, shape)
    lat = np.any([rm[f"{a}_{h}"] for a in LAT_AREAS for h in ("left", "right") if f"{a}_{h}" in rm], 0)
    oro = np.any([rm[f"{a}_{h}"] for a in ORO_AREAS for h in ("left", "right") if f"{a}_{h}" in rm], 0)
    mlag = np.array([int(round(s * fs)) for s in MAP_LAGS_S])
    tl = np.arange(int(round(TR_LAG_S[0] * fs)), int(round(TR_LAG_S[1] * fs)) + 1)
    trc = [cols.index(c) for c in ("orofacial_left", "orofacial_right", "limb_left", "limb_right")]
    out, per_ev = {}, []
    for g in ("orofacial", "limb"):
        gi = cols.index(g)
        for kind, sign in (("pos", 1), ("neg", -1)):
            a = z[f"raw_{kind}"]
            mvp = z[f"move_raw_{kind}"]
            sel = (a[:, 0] == gi) & (a[:, 2] > K * sd[gi]) & np.isfinite(mvp)
            fr_ev, h, eng, mvp = a[sel, 1].astype(int), a[sel, 2] / sd[gi], a[sel, 3].astype(bool), mvp[sel]
            ok = (fr_ev + tl[0] >= 0) & (fr_ev + tl[-1] < T)
            fr_ev, h, eng, mvp = fr_ev[ok], h[ok], eng[ok], mvp[ok]
            hb = np.digitize(h, HBINS[1:-1])
            for c in classes():
                if c[0] != kind:
                    continue
                m = (eng == (c[1] == "eng")) & ((mvp > MOVE_PCTL) == (c[2] == "move")) & (hb == c[3])
                key = f"{g}_{ckey(c)}"
                out[f"n_{key}"] = int(m.sum())
                if not m.any():
                    continue
                idx = fr_ev[m][:, None] + tl[None, :]
                out[f"tr_{key}"] = Xd[trc][:, idx].mean(1).astype(np.float32)
                mv = Vh[fr_ev[m][:, None] + mlag[None, :]].mean(0)                       # nlag x k
                M = 100 * (Ub @ mv.T).T                                                     # nlag x pixels
                if g == "orofacial":
                    out[f"map_{key}"] = np.stack([ds_img(M[j], flat, shape) for j in range(len(mlag))])
                j0, j2 = MAP_LAGS_S.index(0.0), MAP_LAGS_S.index(0.2)
                out[f"spread_{key}"] = np.array([sign * M[j, lat].mean() / (sign * M[j, oro].mean())
                                                 for j in (j0, j2)], np.float32)
            if g == "orofacial":
                iL, iR = cols.index("orofacial_left"), cols.index("orofacial_right")
                w = int(round(0.15 * fs))
                for p, hh, e, mp in zip(fr_ev, h, eng, mvp):
                    aL = sign * Xd[iL, p - w:p + w + 1].max() if sign > 0 else -(-Xd[iL, p - w:p + w + 1]).max()
                    aR = sign * Xd[iR, p - w:p + w + 1].max() if sign > 0 else -(-Xd[iR, p - w:p + w + 1]).max()
                    pL = p - w + int(np.argmax(sign * Xd[iL, p - w:p + w + 1]))
                    pR = p - w + int(np.argmax(sign * Xd[iR, p - w:p + w + 1]))
                    tL = (pL - p) + half_time(Xd[iL], pL, (-int(round(0.3 * fs)),), sign)
                    tR = (pR - p) + half_time(Xd[iR], pR, (-int(round(0.3 * fs)),), sign)
                    per_ev.append((sign, int(e), int(mp > MOVE_PCTL), hh, abs(aL), abs(aR), tL / fs, tR / fs))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, ev=np.array(per_ev, float).reshape(-1, 8), map_lags_s=np.array(MAP_LAGS_S),
                        tr_lags_s=tl / fs, flat=flat, shape=np.array(shape), parea=parea, fs=fs,
                        quiet_eng_min=float((~busy_frames(label, ft, fs)[0] & engaged).sum() / fs / 60), **out)
    return label


# ------------------------------------------------------------------------------------------------------- summary
def load():
    from wfield_local import epochs
    D = {}
    for f in sorted((OUTS / "sessions").glob("*.npz")):
        ep = epochs.epoch_of(f.stem)
        if ep is None or f.stem[-4:] < "0806":
            continue
        D[f.stem] = dict(np.load(f), epoch=ep)
    return D


def class_map(D, an, ep, kind, st, mv, hset, min_n=3):
    """Animal's map for a class, height-matched: mean over height bins (equal weight) of the session-pooled class means."""
    per_h = []
    for h in hset:
        key = f"orofacial_{kind}_{st}_{mv}_h{h}"
        S, N = 0.0, 0
        for k, d in D.items():
            if k.startswith(an) and d["epoch"] == ep and f"map_{key}" in d:
                n = int(d[f"n_{key}"])
                S = S + d[f"map_{key}"] * n
                N += n
        if N >= min_n:
            per_h.append(S / N)
    return np.mean(per_h, 0) if len(per_h) == len(hset) else None


def session_maps(D, an, ep, kind, st, mv, hset, min_n=2):
    """Per-session height-matched maps (for the nested bootstrap), upsampled to full resolution."""
    out = []
    for k, d in sorted(D.items()):
        if not (k.startswith(an) and d["epoch"] == ep):
            continue
        per = []
        for h in hset:
            key = f"orofacial_{kind}_{st}_{mv}_h{h}"
            if f"map_{key}" in d and int(d[f"n_{key}"]) >= min_n:
                per.append(d[f"map_{key}"])
        if len(per) == len(hset):
            out.append(np.mean(per, 0))
    return out


def up(img, shape=(540, 640)):
    full = np.full(shape, np.nan, np.float32)
    u = np.kron(img, np.ones((DS, DS), np.float32))
    full[:u.shape[0], :u.shape[1]] = u
    return full


def fig_maps(D, kind, st_post, hset, mv, out_name, title):
    """Rows: pre (engaged), post class, DELTA with nested-bootstrap outline; columns = map lags."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from quiet_fluct import area_edges, draw_outline
    from wfield_local import beta_maps as bm
    k0 = next(iter(D))
    flat, shape, parea = D[k0]["flat"], tuple(D[k0]["shape"]), D[k0]["parea"]
    edges = area_edges(parea, flat, shape)
    onpix = np.zeros(int(np.prod(shape)), bool)
    onpix[flat] = True
    smask = bm.stat_mask() & onpix.reshape(shape)
    animals = sorted({k[:4] for k in D})
    lags = D[k0]["map_lags_s"]
    pre_st, post_ep = "eng", st_post[1]
    sig = []
    pre = {an: class_map(D, an, "pre", kind, pre_st, mv, hset) for an in animals}
    post = {an: class_map(D, an, post_ep, kind, st_post[0], mv, hset) for an in animals}
    both = [an for an in animals if pre[an] is not None and post[an] is not None]
    if len(both) < 2:
        print(out_name, "skipped: animals with both classes =", both)
        return None
    P = np.mean([pre[a] for a in both], 0)
    Q = np.mean([post[a] for a in both], 0)
    fig, ax = plt.subplots(3, len(lags), figsize=(2.6 * len(lags), 8.4))
    v = np.nanpercentile(np.abs(np.r_[P.ravel(), Q.ravel()]), 99)
    vd = np.nanpercentile(np.abs(Q - P), 99)
    for r, (M, lab) in enumerate(((P, f"pre engaged ({len(both)} an.)"),
                                  (Q, f"{post_ep} {st_post[0]} ({len(both)} an.)"), (Q - P, "DELTA"))):
        for j in range(len(lags)):
            a = ax[r, j]
            im = a.imshow(up(M[j]), cmap="RdBu_r", vmin=-(vd if r == 2 else v), vmax=vd if r == 2 else v)
            if r == 2:
                ps = {x: [up(m[j]) for m in session_maps(D, x, "pre", kind, pre_st, mv, hset)] for x in both}
                qs = {x: [up(m[j]) for m in session_maps(D, x, post_ep, kind, st_post[0], mv, hset)] for x in both}
                ps = {x: y for x, y in ps.items() if y and qs.get(x)}
                qs = {x: qs[x] for x in ps}
                full, lab_t = (bm.significance_contour(ps, qs, method="nested", n_boot=2000, mask=smask,
                                                       normalise=False) if len(ps) >= 2 else (None, "< 2 animals"))
                if full is not None:
                    draw_outline(a, full)
                sig.append(dict(figure=out_name, lag_s=float(lags[j]), animals=len(ps), test=lab_t))
            a.imshow(np.where(edges, 1.0, np.nan), cmap="gray_r", vmin=0, vmax=1.5, alpha=0.5)
            a.set_xticks([])
            a.set_yticks([])
            if r == 0:
                a.set_title(f"{lags[j]:+.1f} s", fontsize=9)
        ax[r, 0].set_ylabel(lab, fontsize=8)
        fig.colorbar(im, ax=ax[r, :].tolist(), fraction=0.02, label="dF/F %")
    fig.suptitle(title + "\nmean of animal means; DELTA outline = nested bootstrap, max-stat p < 0.05, 8x8-px bins",
                 fontsize=9)
    fig.subplots_adjust(top=0.88)
    fig.savefig(OUTS / out_name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return sig


def spread_table(D):
    import pandas as pd
    rows = []
    for k, d in D.items():
        for c in classes():
            key = f"orofacial_{ckey(c)}"
            if f"spread_{key}" in d and int(d[f"n_{key}"]) >= 3:
                s0, s2 = d[f"spread_{key}"]
                rows.append(dict(label=k, animal=k[:4], epoch=d["epoch"], kind=c[0], state=c[1], move=c[2], h=c[3],
                                 n=int(d[f"n_{key}"]), spread0=float(s0), spread02=float(s2)))
    return pd.DataFrame(rows)


def ev_table(D):
    import pandas as pd
    rows = []
    for k, d in D.items():
        E = d["ev"]
        if not len(E):
            continue
        df = pd.DataFrame(E, columns=["sign", "eng", "move", "h", "ampL", "ampR", "tL", "tR"])
        df["label"], df["animal"], df["epoch"] = k, k[:4], d["epoch"]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def summarise():
    import pandas as pd
    pd.set_option("display.width", 220)
    EP = ["pre", "acute", "subacute", "chronic"]
    D = load()
    print(len(D), "sessions")
    S = spread_table(D)
    S.to_csv(OUTS / "spread_sessions.csv", index=False)
    print("\nT1 SPREAD (lateral SSp / orofacial at 0 s), UP events, engaged, height bins 0-1, per animal x epoch, by "
          "movement (session medians):")
    G = S[(S.kind == "pos") & (S.state == "eng") & (S.h <= 1)]
    print(G.pivot_table(index=["move", "animal"], columns="epoch", values="spread0", aggfunc="median")
          .reindex(columns=EP).round(2).to_string())
    print("\nT3 SPREAD, DOWN events, engaged:")
    G = S[(S.kind == "neg") & (S.state == "eng") & (S.h <= 1)]
    print(G.pivot_table(index=["move", "animal"], columns="epoch", values="spread0", aggfunc="median")
          .reindex(columns=EP).round(2).to_string())
    print("\nT4 SPREAD, UP still events, QUIT vs ENGAGED:")
    G = S[(S.kind == "pos") & (S.move == "still") & (S.h <= 1)]
    print(G.pivot_table(index=["state", "animal"], columns="epoch", values="spread0", aggfunc="median")
          .reindex(columns=EP).round(2).to_string())
    print("\nevent counts (engaged, orofacial up, h 0-1) still / move by animal x epoch:")
    print(S[(S.kind == "pos") & (S.state == "eng") & (S.h <= 1)].pivot_table(
        index=["move", "animal"], columns="epoch", values="n", aggfunc="sum").reindex(columns=EP).to_string())
    E = ev_table(D)
    E = E[E.eng == 1]
    E["amp_ratio"] = np.log2(E.ampL / E.ampR)
    E["lag_ms"] = 1000 * (E.tL - E.tR)                                     # < 0: ipsilesional (left) reaches half first
    E.to_csv(OUTS / "events_ipsi_contra.csv", index=False)
    for sg, nm in ((1, "UP"), (-1, "DOWN")):
        G = E[E.sign == sg]
        print(f"\nT2 {nm} events: log2(ipsi / contra amplitude), session medians -> animal x epoch median:")
        print(G.groupby(["label", "animal", "epoch"]).amp_ratio.median().reset_index()
              .pivot_table(index="animal", columns="epoch", values="amp_ratio", aggfunc="median")
              .reindex(columns=EP).round(3).to_string())
        print(f"T2 {nm} events: onset lag ipsi - contra (ms; < 0 = ipsilesional first):")
        print(G.groupby(["label", "animal", "epoch"]).lag_ms.median().reset_index()
              .pivot_table(index="animal", columns="epoch", values="lag_ms", aggfunc="median")
              .reindex(columns=EP).round(1).to_string())
    sig = []
    for kind, mv, st_post, name, ttl in (
            ("pos", "still", ("eng", "acute"), "T1_up_still_pre_vs_acute.png",
             "T1: orofacial UP events WITHOUT face movement, height-matched (4-6 + 6-9 x pre SD): pre vs acute (engaged)"),
            ("pos", "move", ("eng", "acute"), "T1_up_move_pre_vs_acute.png",
             "T1: orofacial UP events WITH face movement, height-matched: pre vs acute (engaged)"),
            ("neg", "still", ("eng", "acute"), "T3_down_still_pre_vs_acute.png",
             "T3: orofacial DOWN events without face movement, height-matched: pre vs acute (engaged)"),
            ("pos", "still", ("quit", "pre"), "T4_up_still_pre_eng_vs_pre_quit.png",
             "T4: orofacial UP events without face movement: pre ENGAGED vs pre QUIT period")):
        r = fig_maps(D, kind, st_post, (0, 1), mv, name, ttl)
        sig += r or []
    pd.DataFrame(sig).to_csv(OUTS / "map_significance.csv", index=False)
    (OUTS / "README.txt").write_text(__doc__)
    print("wrote", OUTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if not a.summarise:
        from wfield_local.parallel import fan_out
        labels = a.only or sorted(f.stem for f in FRV.glob("*.npz") if f.stem[-4:] >= "0806")
        res, fail = fan_out(labels, extract, jobs=a.jobs, label="session")
        print(f"{len(res)} done, {len(fail)} failed")
        for x in fail:
            print("FAILED", x)
        if a.only:
            return
    summarise()


if __name__ == "__main__":
    main()

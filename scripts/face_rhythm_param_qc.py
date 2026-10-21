"""Visual + factor QC of the FaceRhythm tracking-parameter test (`scripts/o2_face_rhythm.py paramtest`; 10-21).

    <fr env> python scripts/face_rhythm_param_qc.py clips     [--bundle fr_cam4_20261021]
    <fr env> python scripts/face_rhythm_param_qc.py decompose [--bundle fr_cam4_20261021] [--rank 10]

Inputs: the bundle's ``out_test/<key>_<set>.npz`` (full-rate points per parameter set), the session's cam4 video (read
with decord, the reader the tracking used) and its v36 cam4 ``windows.npz`` (LP tongue likelihood = licking, nose).
clips      the same frames side by side for every set, each set's mesh drawn on the video, dots coloured by
           |displacement from home| (0-6 px): a 4 s lick bout (most tongue frames) and a 4 s rest window (no tongue,
           least nose motion), written at 50 fps (5x slow) -> <dlc>/face_rhythm/cam4/param_test/clips/.
decompose  per set: the production spectrogram (keyframe notches, VQT 1-60 Hz, 25 Hz) -> 12.5 Hz -> non-negative CP
           (tensorly HALS; FR's TCA, (xy, points) x frequency x time) -> variance explained, and per component its
           spatial map, spectrum and time course against LP tongue visibility and nose speed; components labelled where
           clear (licking: temporal r with tongue; whisking: 6-14 Hz peak on the whisker pads; nose: weight near the
           nose). Two minutes of data: a first look, not the cohort factors.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.o2_face_rhythm import PARAMS, V36  # noqa: E402

FS = PARAMS["fps"]


def _paths(bundle: str):
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    dlc = Path(rv.root("dlc"))
    return rv, dlc / "o2" / bundle, dlc / "face_rhythm" / "cam4" / "param_test"


def load_sets(b: Path) -> tuple[str, dict]:
    sets = {}
    for f in sorted((b / "out_test").glob("*.npz")):
        key, name = f.stem.rsplit("_", 1)
        z = np.load(f)
        sets[name] = {"pts": z["pts"], "home": z["home"], "start": int(z["start"]), "nv": z["nv"]}
    return key, sets


def lp_reference(rv, key: str, start: int, n: int) -> dict:
    """Per frame of the test window: tongue visible (LP likelihood > 0.9) and nose speed (px/frame); NaN outside
    the LP windows."""
    a, d = key.split("_")[:2]
    w = np.load(Path(rv.root("orofacial")) / V36 / "sessions" / a / d / "windows.npz")
    parts = [str(x) for x in w["parts"]]
    src = w["ix_src_frame"].astype(int)
    sel = (src >= start) & (src < start + n)
    fi, first = np.unique(src[sel] - start, return_index=True)
    pose = w["pose"][sel][first]
    tongue = np.full(n, np.nan)
    tongue[fi] = (pose[:, parts.index("tongue"), 2] > 0.9).astype(float)
    nose = np.full((n, 2), np.nan)
    ok = pose[:, parts.index("nose"), 2] > 0.9
    nose[fi[ok]] = pose[ok, parts.index("nose"), :2]
    sp = np.full(n, np.nan)
    sp[1:] = np.hypot(*np.diff(nose, axis=0).T)
    return {"tongue": tongue, "nose_speed": sp, "nose_rest": np.nanmedian(nose, 0)}


def cmd_clips(bundle: str, seconds: float = 4.0, color: str = "disp", only: list[str] | None = None) -> None:
    import cv2
    import decord
    import pandas as pd
    rv, b, out = _paths(bundle)
    key, sets = load_sets(b)
    s0 = next(iter(sets.values()))
    start, n = s0["start"], len(s0["pts"])
    ref = lp_reference(rv, key, start, n)
    L = int(seconds * FS)
    tg = pd.Series(ref["tongue"]).fillna(0).rolling(L).sum().to_numpy()
    ns = pd.Series(np.nan_to_num(ref["nose_speed"], nan=99)).rolling(L).mean().to_numpy()
    lick_end = int(np.nanargmax(tg))
    calm = np.where(tg == 0, ns, np.inf)
    rest_end = int(np.argmin(calm))
    T = pd.read_csv(b / "tasks.tsv", sep="\t", dtype=str)
    r = T[T["key"] == key].iloc[0]
    vr = decord.VideoReader(str(Path(rv.root("behavior_cameras")) / r.date / r.animal / r.video))
    (out / "clips").mkdir(parents=True, exist_ok=True)
    names = [n for n in (only or list(sets)) if n in sets]
    sc = 0.8
    for label, end in (("lick", lick_end), ("rest", rest_end)):
        i0 = end - L + 1
        frames = vr.get_batch(list(range(start + i0, start + end + 1))).asnumpy()
        H, W = frames.shape[1:3]
        h, w_ = int(H * sc), int(W * sc)
        dst = out / "clips" / f"{key}_{label}_{start + i0}{'_plain' if color == 'plain' else ''}{'_' + '-'.join(names) if only else ''}.mp4"
        vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), 50, (w_ * len(names), h + 30))
        for k, fr in enumerate(frames):
            i = i0 + k
            panels = []
            for nm in names:
                im = cv2.resize(fr, (w_, h))
                P, Hm = sets[nm]["pts"][i], sets[nm]["home"]
                if color == "plain":                                  # judge motion by position, not colour
                    col = np.tile(np.array([0, 255, 255], np.uint8), (len(P), 1))
                else:
                    mag = np.clip(np.hypot(*(P - Hm).T) / 6.0, 0, 1)
                    col = cv2.applyColorMap((mag * 255).astype(np.uint8)[:, None],
                                            cv2.COLORMAP_PLASMA)[:, 0, ::-1]   # BGR -> RGB (frames are RGB)
                for (x, y), c in zip(P * sc, col):
                    cv2.circle(im, (int(round(x)), int(round(y))), 1, tuple(int(v) for v in c), -1)
                cv2.putText(im, nm, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                panels.append(im)
            bar = np.zeros((30, w_ * len(names), 3), np.uint8)
            t_ = f"{key}  frame {start + i}  t={i / FS:6.2f} s  tongue={'yes' if ref['tongue'][i] == 1 else '-'}  " \
                 + ("(5x slow)" if color == "plain" else "(dots: |disp| 0-6 px, 5x slow)")
            cv2.putText(bar, t_, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            vw.write(cv2.cvtColor(np.vstack([bar, np.hstack(panels)]), cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"{label}: frames {start + i0}-{start + end} ({int(np.nansum(ref['tongue'][i0:end + 1]))} tongue "
              f"frames) -> {dst}")


def cmd_decompose(bundle: str, rank: int = 10, n_iter: int = 150, threads: int = 6,
                  only: list[str] | None = None) -> None:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import tensorly as tl
    import torch
    from scipy.signal import filtfilt, iirnotch
    from tensorly.decomposition import non_negative_parafac_hals
    import face_rhythm as fr
    torch.set_num_threads(threads)
    rv, b, out = _paths(bundle)
    key, sets = load_sets(b)
    first = next(iter(sets.values()))
    start, n = first["start"], len(first["pts"])
    ref = lp_reference(rv, key, start, n)
    Vq = PARAMS["vqt"]
    vqt = fr.spectral_analysis.VQT_Analyzer(params_VQT=Vq["params_VQT"], batch_size=Vq["batch_size"], device="cpu",
                                            normalization_factor=Vq["normalization_factor"],
                                            spectrogram_exponent=Vq["spectrogram_exponent"],
                                            one_over_f_exponent=Vq["one_over_f_exponent"], verbose=0)
    freqs = np.asarray(vqt.vqt_model.freqs)
    DS, BIN = Vq["params_VQT"]["downsample_factor"], 2                    # 25 Hz -> 12.5 Hz for the CP fit
    nb = n // (DS * BIN)
    tongue = np.nanmean(ref["tongue"][:nb * DS * BIN].reshape(nb, -1), 1)
    nose = np.nanmean(ref["nose_speed"][:nb * DS * BIN].reshape(nb, -1), 1)
    img = None
    mo = np.load(Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "register" / "master_overlay.npz")
    if str(mo["master"]) == "_".join(key.split("_")[:2]):
        img = mo["image"]
    rows = []
    for name, s in sets.items():
        if only and name not in only:
            continue
        pts, home = s["pts"].astype(np.float64), s["home"]
        for fc in PARAMS["notch_hz"]:
            bn, an = iirnotch(fc, PARAMS["notch_q"], FS)
            pts = filtfilt(bn, an, pts, axis=0)
        S, _, _ = vqt.transform(pts.astype(np.float32), home)                # (2, NP, F, T25)
        NP = S.shape[1]
        X = S[..., :nb * BIN].reshape(2 * NP, len(freqs), nb, BIN).mean(-1).astype(np.float32)
        del S, pts
        Xt = tl.tensor(X)
        cp = non_negative_parafac_hals(Xt, rank=rank, n_iter_max=n_iter, init="random", random_state=0)
        Xh = tl.cp_to_tensor(cp)
        ve = 1 - float(((X - Xh) ** 2).sum() / ((X - X.mean()) ** 2).sum())
        A, Fq, Tm = [np.asarray(f) for f in cp.factors]
        A = A * np.asarray(cp.weights)[None]
        share = np.array([(np.linalg.norm(A[:, k]) * np.linalg.norm(Fq[:, k]) * np.linalg.norm(Tm[:, k])) ** 2
                          for k in range(rank)])
        order = np.argsort(share)[::-1]
        sp = np.hypot(A[:NP], A[NP:])                                     # (NP, rank) spatial weight per point
        lat = np.abs(home[:, 0] - ref["nose_rest"][0])
        near_nose = np.hypot(*(home - ref["nose_rest"]).T) < 60
        okb = np.isfinite(tongue) & np.isfinite(nose)
        fig, axs = plt.subplots(rank, 3, figsize=(14, 2.1 * rank), gridspec_kw={"width_ratios": [1, 1, 3]})
        for row, k in enumerate(order):
            pk = float(freqs[np.argmax(Fq[:, k])])
            r_t = float(np.corrcoef(Tm[okb, k], tongue[okb])[0, 1])
            r_n = float(np.corrcoef(Tm[okb, k], nose[okb])[0, 1])
            wsp = sp[:, k] / sp[:, k].sum()
            pad_w = float(wsp[lat > 120].sum())
            nose_w = float(wsp[near_nose].sum())
            label = ("licking" if r_t > 0.4 else "whisking" if 6 <= pk <= 14 and pad_w > 0.5
                     else "nose/sniff" if nose_w > 0.3 else "")
            rows.append({"set": name, "comp": row, "share": share[k] / share.sum(), "peak_hz": pk, "r_tongue": r_t,
                         "r_nose_speed": r_n, "pad_weight": pad_w, "nose_weight": nose_w, "label": label,
                         "var_explained": ve})
            ax = axs[row, 0]
            if img is not None:
                ax.imshow(img, cmap="gray")
            ax.scatter(*home.T, c=sp[:, k], s=1.5, cmap="magma")
            ax.set_axis_off()
            ax.set_title(f"#{row} {label} ({share[k] / share.sum():.0%})", fontsize=8)
            axs[row, 1].semilogx(freqs, Fq[:, k])
            axs[row, 1].set_title(f"peak {pk:.1f} Hz", fontsize=8)
            ax = axs[row, 2]
            tt = np.arange(nb) * DS * BIN / FS
            ax.fill_between(tt, 0, np.nan_to_num(tongue) * Tm[:, k].max(), color="tab:red", alpha=0.2, lw=0)
            ax.plot(tt, Tm[:, k], "k", lw=0.7)
            ax.set_title(f"r tongue {r_t:+.2f}, r nose speed {r_n:+.2f}", fontsize=8)
        axs[-1, 2].set_xlabel("s (red = LP tongue visible)")
        fig.suptitle(f"{key} {name}: non-negative CP rank {rank}, variance explained {ve:.1%}")
        fig.tight_layout()
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"decomp_{key}_{name}.png", dpi=80)
        plt.close(fig)
        np.savez(out / f"decomp_{key}_{name}_factors.npz", spatial=A, spectral=Fq, temporal=Tm, order=order,
                 weights=np.asarray(cp.weights), freqs=freqs, home=home, t_frame=start + np.arange(nb) * DS * BIN,
                 var_explained=ve)
        print(f"{name}: variance explained {ve:.1%}")
    R = pd.DataFrame(rows)
    R.to_csv(out / f"decomp_{key}.csv", index=False, float_format="%.3g")
    print(R.round(2).to_string(index=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["clips", "decompose"])
    ap.add_argument("--bundle", default="fr_cam4_20261021")
    ap.add_argument("--rank", type=int, default=10)
    ap.add_argument("--sets", nargs="*", default=None, help="decompose: only these parameter sets")
    ap.add_argument("--threads", type=int, default=6, help="CPU threads (shared box: keep low)")
    ap.add_argument("--color", choices=["disp", "plain"], default="disp",
                    help="clips: dots coloured by |displacement from home|, or one colour (judge jitter by position)")
    a = ap.parse_args(argv)
    if a.cmd == "clips":
        cmd_clips(a.bundle, color=a.color, only=a.sets)
    else:
        cmd_decompose(a.bundle, a.rank, threads=a.threads, only=a.sets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

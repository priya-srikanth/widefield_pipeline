"""Why is close_R bout-aligned face asymmetry > 0 BEFORE bout onset? (Priya 10-24, open item (a) of the FR handoff.)

`face_rhythm_asym_aligned.py` shows contra - ipsi face asymmetry rising ~0.5 s before the first lick of a bout at close_R,
in every epoch including pre. Two candidate sources, separated here by splitting the bout onsets:

  first      the trial's FIRST bout (its pre-onset window holds cue-evoked / preparatory movement)
  later      every later bout of the trial (its pre-onset window holds the tail of the previous bout)
  leadin     bouts with >= ``--leadin`` s (0.5) without any lick before onset (no carry-over possible)
  carry      bouts with a lick < ``--leadin`` s before onset
  first_late first bouts starting >= 0.6 s after the cue, so the whole -0.5..-0.1 s window is AFTER the cue but before
             any lick (preparation, not the cue response itself)

Per session x class x position: mean contra / ipsi point speed traces (same frozen point PCs and midline as
`face_rhythm_asym_aligned`), -1 .. +1 s around onset; per animal x epoch the mean over sessions; asymmetry = (contra -
ipsi) / (contra + ipsi). Window summaries: PRE -0.5 .. -0.1 s and POST 0 .. +0.5 s.

    python scripts/face_rhythm_bout_split.py [--set master_pre_v1] [--jobs 4] [--plot-only]
-> ``<dlc>/face_rhythm/cam4/readouts/<set>/asym_aligned/bout_split/``: bout_split.png (animal-mean traces; rows =
positions, columns = classes, epochs as greys), bout_split_pre_window.png (per-animal PRE-window asymmetry), data/
bout_split_traces.npz (cache; ``--plot-only`` redraws from it), bout_split_windows.csv (per animal x epoch x class x
position: pre / post asymmetry, n events).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FS = 25.0
LAGS = np.arange(-1.0, 1.0001, 1 / FS)
PRE_WIN, POST_WIN = (-0.5, -0.1), (0.0, 0.5)
CLASSES = ("first", "later", "leadin", "carry", "first_late")
POSITIONS = ("far_L", "close_L", "close_center", "far_center", "close_R", "far_R")   # animal's L = ipsilesional
EPOCHS = ("pre", "acute", "subacute", "chronic")


def bout_classes(L: pd.DataFrame, leadin: float) -> dict[str, np.ndarray]:
    """Boolean masks over ``L`` (lick table, sorted by time) of the bout onsets in each class."""
    t = L.daq_on_s.to_numpy(float)
    gap = np.r_[np.inf, np.diff(t)]                                    # time since the previous lick (any trial)
    onset = (L.n_in_bout == 1).to_numpy() & np.isfinite(L.t_from_cue.to_numpy(float))
    on = L[onset]
    first_bout = on.groupby("cue_idx").bout_idx.transform("min")
    first = np.zeros(len(L), bool)
    first[np.flatnonzero(onset)[(on.bout_idx == first_bout).to_numpy()]] = True
    return {"first": first, "later": onset & ~first, "leadin": onset & (gap >= leadin),
            "carry": onset & (gap < leadin), "first_late": first & (L.t_from_cue.to_numpy(float) >= 0.6)}


def job(item: tuple) -> dict:
    fdir, key, leadin = item
    import face_rhythm_asym_aligned as faa
    from wfield_local import face_rhythm_readouts as frr
    from wfield_local import face_rhythm_validation as frv
    from wfield_local.paths import PathResolver
    fdir = Path(fdir)
    M = np.load(fdir / "master.npz")
    comps = M["points_components"].astype(np.float32)
    contra = M["home"][:, 0] <= float(frr.MASTER_MIDLINE_X)                     # image LEFT = contralesional (cam4)
    with np.load(fdir / "proj" / f"{key}.npz", allow_pickle=True) as z:
        animal, date, epoch = str(z["animal"]), str(z["date"]), str(z["epoch"])
        pcs, t = z["points_pc"].astype(np.float32), z["points_t_daq_s"].astype(float)
        cues = z["cue_daq_s"].astype(float)
    L = frv.lick_table(PathResolver(), animal, date, cues)
    C = bout_classes(L, leadin)
    c, i, _ = faa.side_speed(pcs, t, comps, contra)
    out = {"key": key, "animal": animal, "epoch": epoch, "label": frr._label(animal, date)}
    pos = L.position.astype(str).to_numpy()
    for cl, m in C.items():
        for p in POSITIONS:
            ev = L.daq_on_s.to_numpy(float)[m & (pos == p)]
            out[(cl, p)] = (faa._aligned(t, c, ev, LAGS), faa._aligned(t, i, ev, LAGS), len(ev))
    return out


def save_cache(R: list[dict], f: Path) -> None:
    tr = np.array([[[[r[(cl, p)][0], r[(cl, p)][1]] for p in POSITIONS] for cl in CLASSES] for r in R])
    n = np.array([[[r[(cl, p)][2] for p in POSITIONS] for cl in CLASSES] for r in R])
    np.savez(f, traces=tr, n=n, lags=LAGS, classes=np.array(CLASSES), positions=np.array(POSITIONS),
             **{k: np.array([r[k] for r in R]) for k in ("key", "animal", "epoch", "label")})


def load_cache(f: Path) -> list[dict]:
    z = np.load(f, allow_pickle=True)
    R = []
    for s in range(len(z["key"])):
        r = {k: str(z[k][s]) for k in ("key", "animal", "epoch", "label")}
        for a, cl in enumerate(CLASSES):
            for b, p in enumerate(POSITIONS):
                r[(cl, p)] = (z["traces"][s, a, b, 0], z["traces"][s, a, b, 1], int(z["n"][s, a, b]))
        R.append(r)
    return R


def animal_traces(R: list[dict], min_events: int) -> dict:
    """{(animal, epoch, class, position): (contra, ipsi, n_sessions, n_events)} -- per-session means, a split video's
    halves averaged first (same label), then the mean over sessions."""
    acc: dict = {}
    for r in R:
        for cl in CLASSES:
            for p in POSITIONS:
                c, i, n = r[(cl, p)]
                if n < min_events:
                    continue
                acc.setdefault((r["animal"], r["epoch"], cl, p), {}).setdefault(r["label"], []).append((c, i, n))
    out = {}
    for k, by in acc.items():
        sess = [(np.nanmean([x[0] for x in v], 0), np.nanmean([x[1] for x in v], 0), sum(x[2] for x in v))
                for v in by.values()]
        out[k] = (np.nanmean([s[0] for s in sess], 0), np.nanmean([s[1] for s in sess], 0), len(sess),
                  sum(s[2] for s in sess))
    return out


def asym(c: np.ndarray, i: np.ndarray) -> np.ndarray:
    return (c - i) / (c + i)


def win_mean(y: np.ndarray, w: tuple[float, float]) -> float:
    m = (LAGS >= w[0] - 1e-9) & (LAGS <= w[1] + 1e-9)
    return float(np.nanmean(y[m]))


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import analysis_kit as ak
    from wfield_local import config
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import figure_layout, writeguard
    from wfield_local.epoch_figures import EPOCH_GREY
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--bundle", default=fro.BUNDLE)
    ap.add_argument("--fdir", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--leadin", type=float, default=0.5, help="lick-free time before onset for the 'leadin' class, s")
    ap.add_argument("--min-events", type=int, default=3, help="per session x class x position, else skipped")
    ap.add_argument("--plot-only", action="store_true", help="redraw from data/bout_split_traces.npz")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = a.fdir or fro.factor_dir(rv, a.bundle, a.set)
    out = a.out or Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "readouts" / a.set / "asym_aligned" / "bout_split"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = figure_layout.sidecar_for(out, "bout_split_traces", ".npz")
    if a.plot_only:
        R = load_cache(cache)
    else:
        keys = sorted(f.stem for f in (fdir / "proj").glob("*.npz") if not f.name.endswith(".part.npz"))
        got, fail = ak.fan_sessions([(str(fdir), k, a.leadin) for k in keys], job, jobs=a.jobs, key=lambda it: it[1],
                                    label="bout_split")
        if fail:
            print(f"FAILED {len(fail)}: {fail}")
        R = [r for _, r in got]
        save_cache(R, cache)
    A = animal_traces(R, a.min_events)
    animals = sorted({k[0] for k in A})

    rows = [{"animal": an, "epoch": ep, "class": cl, "position": p, "pre_asym": win_mean(asym(c, i), PRE_WIN),
             "post_asym": win_mean(asym(c, i), POST_WIN), "n_sessions": ns, "n_events": ne}
            for (an, ep, cl, p), (c, i, ns, ne) in A.items()]
    T = pd.DataFrame(rows)
    T.to_csv(out / "bout_split_windows.csv", index=False)
    pd.set_option("display.width", 200)
    for p in ("close_R", "far_R", "close_L"):
        print(f"\n{p}: PRE-window ({PRE_WIN[0]:+g}..{PRE_WIN[1]:+g} s) asymmetry per animal (rule 8), [events]")
        sub = T[T.position == p].copy()
        sub["cell"] = [f"{v:+.3f} [{n}]" for v, n in zip(sub.pre_asym, sub.n_events)]
        print(sub.pivot_table(index=["class", "epoch"], columns="animal", values="cell", aggfunc="first")
              .reindex(pd.MultiIndex.from_product([CLASSES, EPOCHS])).to_string())

    # traces: rows = positions, columns = classes; animal mean, epochs in the repo greys
    fig, axs = plt.subplots(len(POSITIONS), len(CLASSES), figsize=(3.2 * len(CLASSES), 2.2 * len(POSITIONS)),
                            sharex=True, sharey=True, squeeze=False)
    for r_i, p in enumerate(POSITIONS[::-1]):
        for c_i, cl in enumerate(CLASSES):
            ax = axs[r_i, c_i]
            for ep in EPOCHS:
                ys = [asym(*A[(an, ep, cl, p)][:2]) for an in animals if (an, ep, cl, p) in A]
                if len(ys) < 2:
                    continue
                ys = np.array(ys)
                m, se = np.nanmean(ys, 0), np.nanstd(ys, 0) / np.sqrt(len(ys))
                ax.plot(LAGS, m, color=EPOCH_GREY[ep], lw=1.3, label=f"{ep} (n={len(ys)})")
                ax.fill_between(LAGS, m - se, m + se, color=EPOCH_GREY[ep], alpha=0.15, lw=0)
            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(0, color="k", lw=0.5, ls=":")
            ax.axvspan(*PRE_WIN, color="tab:purple", alpha=0.06, lw=0)
            if r_i == 0:
                ax.set_title(cl)
            if c_i == 0:
                ax.set_ylabel(f"{p}\n(c - i) / (c + i)")
            if r_i == len(POSITIONS) - 1:
                ax.set_xlabel("s from bout onset")
            if r_i == 0 and c_i == 0:
                ax.legend(fontsize=6, frameon=False)
    fig.suptitle(f"{fdir.name}: bout-onset face asymmetry split by bout class (mean +- SEM of animals; purple = "
                 f"pre-onset window)")
    fig.tight_layout()
    frf.save(fig, out / "bout_split.png", f"""Bout-onset-aligned face asymmetry (contra - ipsi) /
(contra + ipsi) of mean mesh-point speed, split by bout class: first = the trial's first bout; later = later bouts;
leadin = >= {a.leadin:g} s without a lick before onset; carry = a lick < {a.leadin:g} s before; first_late = first bouts
>= 0.6 s after the cue. Animal means of session means, +- SEM across animals; epochs in grey (light = pre). Purple =
pre-onset window {PRE_WIN[0]:+g}..{PRE_WIN[1]:+g} s. > 0 = more contralesional (image-left) movement.""")
    plt.close(fig)

    # per-animal PRE-window asymmetry, class x epoch, for each position
    fig, axs = plt.subplots(1, len(POSITIONS), figsize=(3.4 * len(POSITIONS), 3.4), sharey=True, squeeze=False)
    for c_i, p in enumerate(POSITIONS[::-1]):
        ax = axs[0, c_i]
        for k, cl in enumerate(CLASSES):
            for e, ep in enumerate(EPOCHS):
                x0 = k * (len(EPOCHS) + 1) + e
                for an in animals:
                    s = T[(T.animal == an) & (T.epoch == ep) & (T["class"] == cl) & (T.position == p)]
                    if len(s):
                        ax.plot(x0, s.pre_asym.iloc[0], "o", ms=3, color=config.animal_color()[an])
                v = T[(T.epoch == ep) & (T["class"] == cl) & (T.position == p)].pre_asym
                if len(v):
                    ax.bar(x0, v.mean(), color=EPOCH_GREY[ep], alpha=0.6, width=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks([k * (len(EPOCHS) + 1) + 1.5 for k in range(len(CLASSES))], CLASSES, rotation=40, fontsize=7)
        ax.set_title(p)
    axs[0, 0].set_ylabel(f"pre-onset asymmetry ({PRE_WIN[0]:+g}..{PRE_WIN[1]:+g} s)")
    for an in animals:
        axs[0, -1].plot([], [], "o", color=config.animal_color()[an], label=an)
    axs[0, -1].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{fdir.name}: pre-onset face asymmetry by bout class (bars = mean of animals, epochs pre -> chronic "
                 f"left to right in each group; dots = animals)")
    fig.tight_layout()
    frf.save(fig, out / "bout_split_pre_window.png", """Pre-onset window asymmetry per animal (dots,
repo animal colours) and the animal mean (bars, epoch greys: pre, acute, subacute, chronic left to right within each
bout class).""")
    plt.close(fig)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

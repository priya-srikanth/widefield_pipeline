"""June as a ~2-month pre-stroke DRIFT FLOOR for the FaceRhythm readouts (11-08/09, Priya: "only as drift maybe").

The existing drift controls compare early vs late AUGUST pre (1-2 weeks apart). June (06/06-06/08) -> August pre spans
~2 months, the time scale of the chronic changes (up to ~day 50). June -> August contains slow drift PLUS training-stage
learning PLUS the June camera set-up, so as a floor it is CONSERVATIVE: a chronic change beyond it is robust; one inside
it is not distinguishable from drift + learning (not "is drift").

Only SCALE-FREE readouts (June has a global amplitude gain, `face_rhythm_june_reference.py`):
  composition  per session x spout side x factor: log2 peak (the 5d quantity) minus the session's mean log2 peak over
               the 12 factors, relative to the animal's own August pre mean (input: june_reference/data/
               june_peak_sessions_side.csv)
  asymmetry    the 5e window asymmetry (contra - ipsi) / (contra + ipsi) of mesh-point speed per alignment (cue / lick /
               bout) x spout position, relative to own pre. Production sessions from asym_aligned/data/asym_traces.pkl
               (no recompute); the 12 June sessions through `face_rhythm_asym_aligned.job` on the June projection set
               (same master point components, same mesh midline).
Per cell: change vs pre for june / acute / subacute / chronic (`frs.epoch_change`: boot_delta, animal-weighted) and
BEYOND THE FLOOR: boot_delta(epoch, june) on the same relative values (paired animals), "beyond" = interval excludes 0
with the same sign as the epoch's change vs pre. Holm per family (composition: 12 factors per side; asymmetry: 6
positions per alignment).

    python scripts/face_rhythm_june_drift.py [--n-boot 10000] [--jobs 2]
-> ``<dlc>/face_rhythm/cam4/readouts/master_pre_v1/june_reference/``: data/june_drift_{composition,asymmetry}.csv,
   june_drift_composition.png, june_drift_asymmetry.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wfield_local.face_rhythm import common as frc  # noqa: E402
from wfield_local.face_rhythm import stats as frs  # noqa: E402

EPOCHS = ("june", "acute", "subacute", "chronic")
JUNE_BUNDLE = "fr_cam4_june_20261108"


def floor_rows(d: pd.DataFrame, value: str, rng, n_boot: int) -> list[dict]:
    """Per post epoch: boot_delta(epoch, june) of ``value`` -> floor_delta / floor_lo / floor_hi / floor_p."""
    from wfield_local import analysis_kit as ak
    june = frs.by_animal(d[d.epoch == "june"], value)
    rows = []
    for ep in EPOCHS[1:]:
        iv, draws = ak.boot_delta(frs.by_animal(d[d.epoch == ep], value), june, rng, n_boot, draws=True)
        rows.append({"epoch": ep, "floor_delta": iv.point if iv else np.nan, "floor_lo": iv.lo if iv else np.nan,
                     "floor_hi": iv.hi if iv else np.nan, "floor_p": ak.boot_p(draws) if iv else np.nan})
    return rows


def cell_stats(d: pd.DataFrame, value: str, rng, n_boot: int, keys: dict) -> list[dict]:
    ch = {r["epoch"]: r for r in frs.epoch_change(d, value, rng, n_boot, epochs=EPOCHS)}
    fl = {r["epoch"]: r for r in floor_rows(d, value, rng, n_boot)}
    return [{**keys, **ch[ep], **fl.get(ep, {})} for ep in EPOCHS]


def finish(T: pd.DataFrame, family: list[str]) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    T = frs.add_holm(T, family)
    T["floor_p_holm"] = np.nan
    m = T.epoch != "june"
    for _, idx in T[m].groupby(family).groups.items():
        T.loc[idx, "floor_p_holm"] = ak.holm(T.loc[idx, "floor_p"].to_numpy(float))
    same = np.sign(T.floor_delta) == np.sign(T.delta)
    excl = (T.floor_lo > 0) | (T.floor_hi < 0)
    T["beyond_floor"] = np.where(m, excl & same & (T["mark"] != ""), False)
    T["beyond_floor_holm"] = T.beyond_floor & (T.floor_p_holm < 0.05)
    return T


def composition(out: Path, rng, n_boot: int) -> pd.DataFrame:
    S = pd.read_csv(out / "data" / "june_peak_sessions_side.csv")
    S["lg"] = np.log2(S.peak.clip(lower=1e-12))
    S["comp"] = S.lg - S.groupby(["label", "group"]).lg.transform("mean")
    S["comp_rel"] = frs.log2_vs_pre(S.assign(v=2.0 ** S.comp), "v", ["animal", "group", "factor"])
    S = S.sort_values(["group", "factor", "animal", "label"], kind="stable")
    rows = []
    for g in ("ipsi", "centre", "contra"):
        for k in sorted(S.factor.unique()):
            d = S[(S.group == g) & (S.factor == k)]
            rows += cell_stats(d, "comp_rel", rng, n_boot, {"group": g, "factor": k, "name": d.name.iloc[0]})
    return finish(pd.DataFrame(rows), ["epoch", "group"])


def asymmetry(out: Path, jdir: Path, rng, n_boot: int, jobs: int, min_events: int = 5) -> pd.DataFrame:
    import face_rhythm_asym_aligned as aa
    from wfield_local import analysis_kit as ak
    R = pd.read_pickle(out.parent / "asym_aligned" / "data" / "asym_traces.pkl")
    cache = out / "data" / "june_asym_traces.pkl"
    if cache.exists():
        RJ = pd.read_pickle(cache)
    else:
        got, fail = ak.fan_sessions([(str(jdir), k) for k in frc.session_keys(jdir)], aa.job, jobs=jobs,
                                    key=lambda it: it[1], label="asym-june")
        assert not fail, fail
        RJ = [{k: v for k, v in r.items() if not (isinstance(k, tuple) and k[0] == "map")} for _, r in got]
        pd.to_pickle(RJ, cache)
    acc = {}
    for r in list(R) + list(RJ):
        for al in aa.ALIGN:
            for p in aa.POSITIONS:
                cc, ii, n = r[(al, p)]
                if n >= min_events:
                    acc.setdefault((r["animal"], r["epoch"], al, p, r["label"]), []).append((cc, ii))
    srows = []
    for (an, ep, al, p, lab), v in acc.items():
        cc, ii = np.nanmean([x[0] for x in v], 0), np.nanmean([x[1] for x in v], 0)
        lags, win = aa.ALIGN[al][:2]
        w = (lags >= win[0]) & (lags <= win[1])
        cw, iw = float(np.nanmean(cc[w])), float(np.nanmean(ii[w]))
        srows.append({"animal": an, "epoch": ep, "label": lab, "align": al, "position": p,
                      "asym": (cw - iw) / (cw + iw)})
    Ss = pd.DataFrame(srows).sort_values(["align", "position", "animal", "label"], kind="stable")
    Ss["rel"] = Ss.asym - Ss.groupby(["animal", "align", "position"]).asym.transform(
        lambda x: x[Ss.loc[x.index, "epoch"] == "pre"].mean())
    Ss.to_csv(out / "data" / "june_drift_asymmetry_sessions.csv", index=False, float_format="%.5g")
    rows = []
    for al in aa.ALIGN:
        for p in aa.POSITIONS:
            rows += cell_stats(Ss[(Ss["align"] == al) & (Ss.position == p)], "rel", rng, n_boot,
                               {"align": al, "position": p})
    return finish(pd.DataFrame(rows), ["epoch", "align"])


def figure(T: pd.DataFrame, cols: list[str], rows_key: str, row_labels: list[str], out_png: Path, title: str,
           xlabel: str, caption: str) -> None:
    from wfield_local import face_rhythm_figures as frf
    ec = {**frc.epoch_colors(), "june": "0.55"}
    with frf.style():
        fig, axs = frf.subplots(1, len(cols), figsize=(3.9 * len(cols) + 2.0, 0.42 * len(row_labels) + 1.8),
                                squeeze=False, sharey=True)
        for j, c in enumerate(cols):
            ax = axs[0, j]
            d0 = T[T.col == c]
            for i, ep in enumerate(("june", "acute", "chronic")):
                d = d0[d0.epoch == ep].set_index(rows_key).reindex(row_labels)
                y = np.arange(len(row_labels)) + (i - 1) * 0.25
                ax.errorbar(d.delta, y, xerr=[d.delta - d.ci_lo, d.ci_hi - d.delta], fmt="o", ms=3.5, lw=0.9,
                            color=ec[ep], mfc="white" if ep == "june" else ec[ep], label=ep)
                if ep != "june":
                    for yy, x, b, bh in zip(y, d.ci_hi, d.beyond_floor, d.beyond_floor_holm):
                        if b:
                            ax.text(x + 0.02, yy, "##" if bh else "#", va="center", fontsize=7, color=ec[ep])
            jd = d0[d0.epoch == "june"].set_index(rows_key).reindex(row_labels)
            ax.barh(np.arange(len(row_labels)), jd.ci_hi - jd.ci_lo, left=jd.ci_lo, height=0.8, color="0.85",
                    zorder=0, lw=0)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_title(c, fontsize=10)
            ax.set_xlabel(xlabel)
        axs[0, 0].set_yticks(range(len(row_labels)))
        axs[0, 0].set_yticklabels(row_labels, fontsize=8)
        axs[0, 0].invert_yaxis()
        frf.legend_outside(axs[0, -1])
        fig.suptitle(title)
        frf.save(fig, out_png, caption)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import o2_inference as oi
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    import face_rhythm_asym_aligned as aa
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args(argv)
    rv = PathResolver()
    out = frc.readouts_dir(fro.SET, rv) / "june_reference"
    writeguard.assert_writable(out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    jdir = oi.bundle_dir(rv, JUNE_BUNDLE) / "factors" / f"{fro.SET}_june"
    pd.set_option("display.width", 250)

    rng = np.random.default_rng(0)
    C = composition(out, rng, a.n_boot)
    C.to_csv(out / "data" / "june_drift_composition.csv", index=False, float_format="%.4g")
    rng = np.random.default_rng(1)
    A = asymmetry(out, jdir, rng, a.n_boot, a.jobs)
    A.to_csv(out / "data" / "june_drift_asymmetry.csv", index=False, float_format="%.4g")

    for nm, T, idx in (("COMPOSITION", C, ["group", "factor", "name"]), ("ASYMMETRY", A, ["align", "position"])):
        T = T.assign(cell=T.delta.round(2).astype(str) + T["mark"].fillna("")
                     + np.where(T.beyond_floor_holm, "##", np.where(T.beyond_floor, "#", "")))
        print(f"\n{nm}: change vs own August pre; * CI excl 0, ** + Holm; # beyond the June floor, ## + Holm")
        print(T.pivot_table(index=idx, columns="epoch", values="cell", aggfunc="first")
              .reindex(columns=list(EPOCHS)).to_string())

    names = [f"{k} {n}" for k, n in C.groupby("factor").name.first().items()]
    Cf = C.assign(col=C.group.map({"ipsi": "ipsi spouts", "centre": "center spouts", "contra": "contra spouts"}),
                  row=[f"{k} {n}" for k, n in zip(C.factor, C.name)])
    common = ("June (06/06-06/08, 3 sessions per animal, never in the fit) projected on the frozen August-pre factor "
              "set master_pre_v1; cam4 FaceRhythm, engaged trials. Dot = change vs the animal's OWN August pre (paired "
              f"animals -> sessions bootstrap, animal-weighted, {a.n_boot} draws), bar = 95 % interval; gray band = "
              "June's interval = the ~2-month pre-stroke floor (slow drift + training stage + camera set-up). "
              "# = the epoch differs from June by the same bootstrap (interval excludes 0, same sign as its change vs "
              "pre, and its own change passes); ## = also Holm.")
    figure(Cf, ["ipsi spouts", "center spouts", "contra spouts"], "row", names, out / "june_drift_composition.png",
           "FaceRhythm factor composition: acute / chronic vs a June -> August drift floor",
           "log2 composition change vs own pre",
           "Composition = log2 PEAK (cue 0..+4 s, trial-averaged absolute expression) minus the session's mean log2 "
           "peak over the 12 factors (scale-free: removes June's global gain). Holm over 12 factors per side. " + common)
    lab = {p: aa.POS_LABEL[p].replace(chr(10), " ") for p in aa.POSITIONS}
    Af = A.assign(col=A["align"] + "-aligned", row=A.position.map(lab))
    rows = [lab[p] for p in frc.POSITIONS_DISPLAY]          # canonical spout order (Priya 11-03)
    figure(Af, [f"{al}-aligned" for al in aa.ALIGN], "row", rows, out / "june_drift_asymmetry.png",
           "Contra / ipsi face-movement asymmetry: acute / chronic vs a June -> August drift floor",
           "asymmetry change vs own pre",
           "Asymmetry = (contra - ipsi) / (contra + ipsi) mesh-point speed (frozen point PCs; cam4 contralesional = "
           "image left of the master nose midline) in the window (cue 0..+2 s, LP lick -0.1..+0.2 s, bout 0..+0.5 s), "
           "per spout position; > 0 = more contralesional. Holm over 6 positions per alignment. " + common)
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

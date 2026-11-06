"""2-D front camera (cam4) vs triangulated 3-D tongue: per-session agreement and target decoding.

Replaces the per-animal table on the PI-deck slide "Validation: the 2-D front camera agrees with 3-D; 3-D sees more"
(kinematics page section 10) with one open marker per session (shape = animal, colour = epoch).

Sources (DeepLabCut/Widefield/session_poses/pose3d/, read as is):
  crosscheck/agreement.csv      per session: r | lateral: cam4 lr vs -ml ; r | peak angle vs -ml   (125 sessions)
  crosscheck/jaw_agreement.csv  per session: r_opening (jaw opening per trial, cam4 px vs 3-D mm) (107 sessions; no June)
  vs_cam4/decoding.csv          per ANIMAL only: target spout from the lick endpoint, 6 positions, August pre,
                                leave one session out, balanced accuracy pooled over held-out sessions.
  Per-SESSION decoding is not saved by scripts/pose3d_vs_cam4.py, so it is recomputed here with the SAME code path
  (same licks: kinematics/licks.csv.gz joined to v36 cam4 licks.csv; same features, model, folds): the accuracy of
  each held-out session's own licks. The pooled per-animal value is recomputed too and checked against decoding.csv.

Cohort 1 only (PS92-95). Epoch = `epoch_figures.epoch_of_day` on days from `config.stroke_date` (checked against the
source epoch column). Per-animal median over sessions (left) and pooled per-animal accuracy (right) are the slide
table's numbers; checked below.

Run:  python fig_pose_validation_sessions.py   (locanmf env)
Out:  build_inputs/pose_validation_sessions.png / .csv

PROVENANCE: recreated 2026-11-04 (RECONCILIATION section 7) from the share-only
  M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-02/fig_pose_validation_sessions.py. Computation, layout and
  styling unchanged. Changes: the pose / v36 inputs are read in place through the path resolver
  (dlc root, orofacial root + the adopted run name from orofacial_cohort.RUN; tracking outputs, not rebuild-dependent);
  --out / --plot-only / figdata (`_common`); no sys.path pin of the main checkout.

    PYTHONPATH=<wf_rebuild> python scripts/rebuild_figs/fig_pose_validation_sessions.py [--out DIR] [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wfield_local import config  # noqa: E402
from wfield_local.epoch_figures import epoch_of_day  # noqa: E402
from wfield_local.orofacial_cohort import EPOCH_LINE, RUN  # noqa: E402

import _common  # noqa: E402

NAME = "pose_validation_sessions"
# outputs: --out, default _common.default_out() = run_mode.subdir(<rebuild_root>/rebuild_figs) (routed, coordinator 11-04)
P3 = Path(config.resolver().root("dlc")) / "session_poses" / "pose3d"            # read in place
V36 = Path(config.resolver().root("orofacial")) / RUN / "sessions"              # read in place
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
MARK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
EPOCHS = ("pre", "acute", "subacute", "chronic")
POS = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")
FEATS = {"cam4": ["lr_px", "ap_px"], "3-D": ["side_ml", "out_ap", "down_dv"]}   # = pose3d_vs_cam4 "cam4 2-D", "3-D"
CHANCE = 1 / 6
AGREE = (("lateral", "Lateral\nposition"), ("angle", "Peak angle\nvs 3-D lateral"), ("jaw", "Jaw opening\nper trial"))
TABLE = {"lateral": (.97, .94, .97, .97), "angle": (.87, .69, .87, .86), "jaw": (.64, .57, .67, .83),
         "dec_cam4": (.53, .42, .78, .59), "dec_3-D": (.81, .67, .86, .74)}


def _day(animal: str, ymd: str) -> int:
    s = config.stroke_date(animal)
    return (date(2026, int(ymd[4:6]), int(ymd[6:8])) - date(2026, int(s[:2]), int(s[2:]))).days


def _epoch(D: pd.DataFrame) -> pd.DataFrame:
    D = D.copy()
    D["animal"] = D.session.str[:4]
    D["date"] = D.session.str[5:]
    D = D[D.animal.isin(ANIMALS)]
    D["day"] = [_day(a, d) for a, d in zip(D.animal, D.date)]
    D["epoch_fig"] = [epoch_of_day(a, d) for a, d in zip(D.animal, D.day)]
    bad = D[D.epoch_fig != D.epoch]
    if len(bad):
        print(f"  WARNING epoch mismatch vs source:\n{bad[['session', 'day', 'epoch_fig', 'epoch']]}")
    D["epoch"] = D.epoch_fig
    return D.drop(columns="epoch_fig")


def load_agreement() -> pd.DataFrame:
    A = pd.read_csv(P3 / "crosscheck" / "agreement.csv").rename(
        columns={"r | lateral: cam4 lr vs -ml": "lateral", "r | peak angle vs -ml": "angle"})
    from wfield_local import trial_classes as tc               # jaw agreement = pose3d_jaw_crosscheck's routed output (11-06)
    J = pd.read_csv(tc.kin_out_dir(P3, "crosscheck", script="pose3d_jaw_crosscheck") / "jaw_agreement.csv").rename(
        columns={"r_opening": "jaw"})
    S = A[["session", "epoch", "lateral", "angle"]].merge(J[["session", "jaw"]], on="session", how="outer")
    S["epoch"] = S.epoch.fillna(S.session.map(J.set_index("session").epoch))
    return _epoch(S).sort_values(["animal", "day"]).reset_index(drop=True)


def load_licks() -> pd.DataFrame:
    """Same lick set as scripts/pose3d_vs_cam4.load (August pre only, to keep it small): since 11-06 (A20) licks of
    camera-working-period trials only (`trial_classes.gate_kin_licks`; WFIELD_KIN_GATE=legacy = every post-cue lick)."""
    from wfield_local import trial_classes as tc
    K = pd.read_csv(P3 / "kinematics" / "licks.csv.gz", dtype={"date": str})
    K = tc.gate_kin_licks(K[K.animal.isin(ANIMALS) & (K.epoch == "pre") & ~K.date.map(config.early_pre)])
    out = []
    for s, g in K.groupby("session"):
        a, d = s.split("_")
        C = pd.read_csv(V36 / a / d / "licks.csv",
                        usecols=["trial_id", "lick_idx", "lr_px", "ap_px", "protrusion_px", "peak_angle_deg"])
        out.append(g.merge(C, on=["trial_id", "lick_idx"], how="inner"))
    return pd.concat(out, ignore_index=True)


def decode_sessions(L: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-session-out exactly as pose3d_vs_cam4.decode; keep per held-out session AND pooled accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows = []
    for animal, A in L.groupby("animal"):
        for name, f in FEATS.items():
            D = A[A.position.isin(POS)].dropna(subset=f)
            yt, yp = [], []
            for s in D.session.unique():
                tr, te = D[D.session != s], D[D.session == s]
                if tr.position.nunique() < len(POS) or not len(te):
                    continue
                m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr[f], tr.position)
                p = list(m.predict(te[f]))
                yt += list(te.position)
                yp += p
                rows.append({"animal": animal, "session": s, "features": name, "n_licks": len(te),
                             "acc": balanced_accuracy_score(te.position, p)})
            rows.append({"animal": animal, "session": "POOLED", "features": name, "n_licks": len(yt),
                         "acc": balanced_accuracy_score(yt, yp)})
    return pd.DataFrame(rows)


def check(S: pd.DataFrame, Dec: pd.DataFrame) -> None:
    from wfield_local import trial_classes as tc
    ref = pd.read_csv(tc.kin_out_dir(P3, "vs_cam4") / "decoding.csv").set_index(["animal", "features"])["6 positions"]
    for i, a in enumerate(ANIMALS):
        for k in ("lateral", "angle", "jaw"):
            v = S.loc[S.animal == a, k].median()
            flag = "" if abs(v - TABLE[k][i]) <= 0.005 else "   <-- TABLE MISMATCH"
            print(f"  {a} {k:8s} median {v:.3f}  table {TABLE[k][i]:.2f}{flag}")
        for name, refname in (("cam4", "cam4 2-D"), ("3-D", "3-D")):
            v = Dec[(Dec.animal == a) & (Dec.features == name) & (Dec.session == "POOLED")].acc.iloc[0]
            flag = "" if abs(v - TABLE[f"dec_{name}"][i]) <= 0.005 else "   <-- TABLE MISMATCH"
            print(f"  {a} dec {name:4s} pooled {v:.3f}  decoding.csv {ref[(a, refname)]:.3f}  "
                  f"table {TABLE[f'dec_{name}'][i]:.2f}{flag}")


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    a = ap.parse_args(argv)
    OUT = a.out
    if a.plot_only:
        S, Dec = _common.load_figdata(OUT, NAME)
    else:
        S, Dec = load_agreement(), decode_sessions(load_licks())
        _common.save_figdata(OUT, NAME, (S, Dec))
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    check(S, Dec)
    OUT.mkdir(parents=True, exist_ok=True)
    long = S.melt(id_vars=["animal", "session", "date", "day", "epoch"], value_vars=["lateral", "angle", "jaw"],
                  var_name="measure", value_name="value").dropna(subset=["value"])
    long["measure"] = long.measure.map({"lateral": "agreement r: lateral position", "angle":
                                        "agreement r: peak angle vs 3-D lateral", "jaw": "agreement r: jaw opening per trial"})
    d = Dec.copy()
    d["measure"] = "target decoding balanced acc (" + d.features + ")" + np.where(d.session == "POOLED", " pooled", "")
    d["epoch"] = np.where(d.session == "POOLED", "pre (August)", "pre")
    d["date"] = d.session.str[5:]
    d = d.rename(columns={"acc": "value"})
    pd.concat([long, d[["animal", "session", "date", "epoch", "measure", "value", "n_licks"]]], ignore_index=True).to_csv(
        OUT / "pose_validation_sessions.csv", index=False, float_format="%.4f")

    rng = np.random.default_rng(0)
    off = {a: o for a, o in zip(ANIMALS, (-0.27, -0.09, 0.09, 0.27))}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.45, 1]})
    # left: agreement r per session
    for i, (k, _) in enumerate(AGREE):
        for a in ANIMALS:
            g = S[(S.animal == a)].dropna(subset=[k])
            x = i + off[a] + rng.uniform(-0.04, 0.04, len(g))
            a1.scatter(x, g[k], marker=MARK[a], s=22, facecolors="none",
                       edgecolors=[EPOCH_LINE[e] for e in g.epoch], linewidths=0.9, alpha=0.8, zorder=2)
            a1.plot([i + off[a] - 0.075, i + off[a] + 0.075], [g[k].median()] * 2, color="0.1", lw=2.4,
                    solid_capstyle="butt", zorder=3)
    a1.set_xticks(range(len(AGREE)))
    a1.set_xticklabels([t for _, t in AGREE], fontsize=11)
    a1.set_xlim(-0.5, len(AGREE) - 0.5)
    a1.set_ylim(0, 1.02)
    a1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a1.set_ylabel("cam4 vs 3-D, r (per session)", fontsize=12)
    a1.set_title("Per-lick agreement, cam4 (2-D) vs 3-D", fontsize=12, pad=6)
    # right: decoding, cam4 vs 3-D
    a2.axhline(CHANCE, color="0.45", ls="--", lw=1.1, zorder=1)
    xc = {"cam4": 0, "3-D": 1}
    for a in ANIMALS:
        pooled = []
        for name in ("cam4", "3-D"):
            g = Dec[(Dec.animal == a) & (Dec.features == name)]
            ps = g[g.session != "POOLED"]
            x = xc[name] + off[a] + rng.uniform(-0.03, 0.03, len(ps))
            a2.scatter(x, ps.acc, marker=MARK[a], s=22, facecolors="none", edgecolors=EPOCH_LINE["pre"],
                       linewidths=0.9, alpha=0.7, zorder=2)
            v = g[g.session == "POOLED"].acc.iloc[0]
            a2.plot([xc[name] + off[a] - 0.075, xc[name] + off[a] + 0.075], [v] * 2, color="0.1", lw=2.4,
                    solid_capstyle="butt", zorder=3)
            pooled.append(v)
        a2.plot([off[a] + 0.075, 1 + off[a] - 0.075], pooled, color="0.35", lw=1.0, zorder=2)   # cam4 -> 3-D
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["cam4 (2-D)", "3-D"], fontsize=11)
    a2.set_xlim(-0.5, 1.5)
    a2.set_ylim(0, 1.02)
    a2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a2.set_ylabel("Target decoding accuracy", fontsize=12)
    a2.set_title("Target spout from lick endpoint (Aug pre)", fontsize=12, pad=6)
    a2.text(1.48, CHANCE - 0.015, "chance", ha="right", va="top", fontsize=11, color="0.4")

    h = [Line2D([], [], ls="none", marker=MARK[a], mfc="none", mec="0.3", ms=7, label=a) for a in ANIMALS]
    h += [Line2D([], [], ls="none", marker="o", mfc="none", mec=EPOCH_LINE[e], mew=1.6, ms=7,
                 label=e.capitalize()) for e in EPOCHS]
    h += [Line2D([], [], color="0.1", lw=2.4, label="Per-animal value")]
    fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=1.0, handlelength=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.5)
    save_png_svg(fig, OUT / "pose_validation_sessions.png", dpi=200, facecolor="white")
    print("sessions per animal (agreement lateral/angle, jaw; decoding):")
    print(S.groupby("animal")[["lateral", "angle", "jaw"]].count().to_string())
    print(S.groupby(["animal", "epoch"]).size().unstack().to_string())
    print(Dec[Dec.session != "POOLED"].groupby(["animal", "features"]).size().unstack().to_string())
    print(f"-> {OUT / 'pose_validation_sessions.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

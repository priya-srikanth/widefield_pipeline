"""BEHAVIOUR-ONLY decoder of lick position (overnight queue 5f, approved LOW item; SVD_GAUSS queue E8 "behaviour-only
decoder from FaceRhythm (+ 3-D kinematics): same targets, compare with the cortical decoder"). Is the cortical lick-position
information MORE than what the face / tongue dynamics carry?

Same licks, targets (lateral mm, angle deg) and frozen scheme as the KEY cortical where-decoder (lick_refit_decoders
"frozen": per animal x lick set, ridge = StandardScaler + RidgeCV per target, trained on the animal's AUGUST PRE licks,
pre sessions leave-one-session-out, post sessions scored by the all-pre fit). Feature sets (rows), all scored on ONE
common lick set per session (every feature below finite), so differences are the features, not the licks:
  cortex              the KEY features: Allen areas + ALM / tjM1 split (ROI v8), / frame-wise session SD, lags 0..0.2 s
  behaviour           FaceRhythm + kinematics (below)
  FaceRhythm          the five master_pre_v1 mode-A factors of the frozen encoder (iti_licks.FR_KEEP: lick, right, left,
                      close, far; COMBINED_ENCODING_FR_GROUPS=master_v1:A, absolute expression), each averaged in THREE
                      windows around the lick peak, -0.1..0, 0..0.1, 0.1..0.2 s (15 features). Read UNBLANKED (blank=False)
                      as in 5a (lick_covariate_reference): the frozen encoder blanks FR outside spout strobe .. trial end,
                      which removes ~96 % of ITI lick peaks; inside the trial windows the values equal the encoder's (25 Hz),
                      in the ITI they are the projection's 5 Hz session samples (smoother; caveat). `fr_in_mask` = share
                      the frozen mask would have kept.
  kinematics          3-D reach (reach_mm, tip distance from the mouth at the peak) and peak tongue speed (speed_peak)
  cortex + behaviour  both (does cortex add to behaviour: r(cortex + behaviour) - r(behaviour))
NO LEAKAGE of the target: the kinematic features carry no lateral sign and no direction -- reach and speed are unsigned
magnitudes. EXCLUDED on purpose: side_ml / lateral / angle (the targets), every *_target / *_err / miss / midline / frontal
column, the landing spout / position, ELEVATION (atan2(down, hypot(lateral, forward)) has |lateral| in it), and out_ap /
down_dv / sagittal extension (with reach they give |lateral| = sqrt(reach^2 - sagittal^2) exactly). Reach alone still
correlates with |lateral| through the spout geometry (far spouts are both farther and more lateral); the `kinematics` row
measures how much that buys (its r is the leakage ceiling of the kinematic part). FaceRhythm factors are face MOTION
(mesh displacement around the mouth, mode A); their left / right factors carry direction legitimately -- that is the
question: how much of the lick position is visible in the face.
Readouts per session: r (decoded vs actual), r / animal's pre mean (the row's own pre). Change vs pre per row and PAIRED
contrasts per session (cortex - behaviour; cortex + behaviour - behaviour), then post - pre: nested bootstrap
(analysis_kit.boot_delta, lick_nonlinear_decoder.tests); held-out pre r levels and their paired differences per animal.
Outputs <OUT_W>/behaviour_decoder_{features.csv.gz,sessions.csv,tests.csv,pre.csv,coverage.csv},
figures/lick_behaviour_decoder.png, figdata/lick_behaviour_decoder.pkl (--plot-only redraws from it).
Usage: python scripts/lick_behaviour_decoder.py [--plot-only] [--animals PS94 ...] [--jobs 8]   (frozen env required)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import lick_nonlinear_decoder as nl  # noqa: E402
import lick_refit_decoders as lr  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN, EP, SETS, TARGETS = nl.AN, nl.EP, nl.SETS, nl.TARGETS
FR_GRID = np.round(np.arange(-0.1, 0.2001, 0.02), 3)
WINS = {"m100_0": (-0.1, -0.001), "0_100": (0.0, 0.099), "100_200": (0.1, 0.2)}    # grid points inside each window
KIN = ["reach_mm", "speed_peak"]
FRF = [f"{v}_{w}" for v in il.FR_KEEP for w in WINS]
ROWS = {"cortex": None, "behaviour": FRF + KIN, "FaceRhythm": FRF, "kinematics": KIN, "cortex + behaviour": FRF + KIN}
RCOL = {"cortex": "#7b3294", "behaviour": "#1b9e77", "FaceRhythm": "#66c2a5", "kinematics": "#a6761d",
        "cortex + behaviour": "#2c7fb8"}
CONTRASTS = (("cortex", "behaviour"), ("cortex + behaviour", "behaviour"))
STEM = "lick_behaviour_decoder"


# ------------------------------------------------------------------------------------------------ per-lick features
def _session_beh(label: str) -> pd.DataFrame:
    """One session (module-level for spawn) -> per lick of the session file (both sets): set, label, row, t, KIN, FRF,
    fr_in_mask. Kinematics joined on lick_uid -> exact DAQ peak time (as iti_where_decoder.coords)."""
    from wfield_local import config
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    d = il._load(label)
    A = pd.read_csv(oo.session_dir(rv, oc.RUN, an, date) / "licks_all.csv", usecols=["daq_peak_s"])
    K = il.lick_table()
    K = K[K.session == f"{an}_{date}"].drop_duplicates("lick_uid")
    kt = pd.DataFrame({c: K[c].to_numpy(float) for c in KIN}, index=A.daq_peak_s.to_numpy(float)[K.lick_uid.to_numpy(int)])
    kt = kt[~kt.index.duplicated()]
    out = []
    for st, tk in (("iti", "iti_t"), ("trial", "tr_t")):
        t = d[tk].astype(float)
        R = pd.DataFrame({"set": st, "label": label, "row": np.arange(len(t)), "t": t})
        R[KIN] = kt.reindex(t)[KIN].to_numpy()
        try:
            grid = (t[:, None] + FR_GRID[None, :]).ravel()
            FU = il.fr_unblanked(an, date, grid, rv)     # the shared UNBLANKED loader (rule 9; 10-28)
            for j, v in enumerate(il.FR_KEEP):
                V = FU[:, j].reshape(len(t), len(FR_GRID))
                for w, (lo_, hi_) in WINS.items():
                    W = V[:, (FR_GRID >= lo_) & (FR_GRID <= hi_)]
                    ok = np.isfinite(W).sum(1) >= W.shape[1] / 2
                    with np.errstate(all="ignore"):
                        R[f"{v}_{w}"] = np.where(ok, np.nanmean(W, 1), np.nan)
            R["fr_in_mask"] = fro.keep_mask(an, date, rv, t)
        except FileNotFoundError as ex:                  # no FaceRhythm for this session: NaN (reported in coverage)
            print(f"  {label}: no FaceRhythm ({ex})", flush=True)
            R[FRF] = np.nan
            R["fr_in_mask"] = False
        out.append(R)
    return pd.concat(out, ignore_index=True)


def features(labels: list[str], jobs: int | None) -> pd.DataFrame:
    """Per-lick behaviour features of every session file lick (cached <OUT_W>/behaviour_decoder_features.csv.gz)."""
    f = il.OUT_W / "behaviour_decoder_features.csv.gz"
    if f.exists():
        C = pd.read_csv(f)
        if set(labels) <= set(C.label):
            return C
    res, fail = ak.fan_sessions(sorted(labels), _session_beh, jobs=jobs, label="behaviour features")
    if fail:
        raise RuntimeError(f"behaviour features failed for {len(fail)} sessions: {fail[:3]}")
    C = pd.concat([v for _it, v in res], ignore_index=True)
    f.parent.mkdir(parents=True, exist_ok=True)
    C.to_csv(f, index=False)
    return C


# ------------------------------------------------------------------------------------------------ decoders
def compute(animals: list[str] | None, jobs: int | None) -> dict:
    D = lr.load_z()
    if animals:
        D = {x: d for x, d in D.items() if d["animal"] in animals}
    C = features(sorted(D), jobs).set_index(["set", "label", "row"]).sort_index()
    cols = list(TARGETS.values())
    rows, cov = [], []
    for st in SETS:
        for a in AN:
            S = {}
            for x, d in sorted(D.items()):
                if d["animal"] != a:
                    continue
                X, Y = d[st]
                B = C.loc[(st, x)].reindex(d[f"{st}_idx"])
                Bf = B[FRF + KIN].to_numpy(float)
                ok = np.isfinite(Bf).all(1)
                cov.append(dict(set=st, animal=a, label=x, epoch=d["epoch"], n=len(Y), common=float(ok.mean()),
                                fr_finite=float(np.isfinite(B[FRF].to_numpy(float)).all(1).mean()),
                                kin_finite=float(np.isfinite(B[KIN].to_numpy(float)).all(1).mean()),
                                fr_in_mask=float(B.fr_in_mask.astype(float).mean())))
                if ok.sum() >= lr.MIN_LICKS:
                    S[x] = (d["epoch"], {"cortex": X[ok], "beh": B[ok]}, Y[ok][:, cols])
            pre = [x for x, v in S.items() if v[0] == "pre"]
            if len(pre) < 3:
                continue
            for row, fc in ROWS.items():
                def feats(x, fc=fc, row=row, S=S):
                    Xc, Bd = S[x][1]["cortex"], S[x][1]["beh"]
                    if fc is None:
                        return Xc
                    Xb = Bd[fc].to_numpy(float)
                    return np.hstack([Xc, Xb]) if row == "cortex + behaviour" else Xb
                fits = {h: lr._fit(np.vstack([feats(q) for q in pre if q != h]), np.vstack([S[q][2] for q in pre if q != h]))
                        for h in [*pre, None]}
                for x, (ep, _f, Y) in S.items():
                    P = lr._pred(fits[x if ep == "pre" else None], feats(x))
                    for j, tg in enumerate(TARGETS):
                        rows.append(dict(set=st, animal=a, label=x, epoch=ep, model=row, target=tg, n=len(Y),
                                         r=float(np.corrcoef(P[:, j], Y[:, j])[0, 1])))
            print(f"[{st} {a}] {len(pre)} pre sessions, {sum(len(v[2]) for v in S.values())} common licks", flush=True)
    S = pd.DataFrame(rows)
    pm = S[S.epoch == "pre"].groupby(["set", "animal", "model", "target"]).r.mean()
    S["r_norm"] = S.r / [pm.get(k, np.nan) for k in zip(S.set, S.animal, S.model, S.target)]
    T = nl.tests(S, models=tuple(ROWS), contrasts=CONTRASTS)
    PL = nl.pre_levels(S, models=tuple(ROWS), contrasts=CONTRASTS)
    return dict(S=S, T=T, PL=PL, coverage=pd.DataFrame(cov), animals=sorted(S.animal.unique()))


# ------------------------------------------------------------------------------------------------ figure
def figure_data(D: dict) -> dict:
    S = D["S"]
    lines = {}
    for (st, tg, m), g in S.groupby(["set", "target", "model"]):
        for val in ("r", "r_norm"):
            am = g.groupby(["animal", "epoch"])[val].mean().unstack("epoch").reindex(columns=EP)
            lines[(st, tg, m, val)] = (am.mean().to_numpy(), (am.std(ddof=1) / np.sqrt(am.count())).to_numpy())
    cv = D["coverage"].groupby(["set", "epoch"])[["common", "fr_in_mask"]].mean()
    return dict(lines=lines, T=D["T"], PL=D["PL"], coverage=cv, animals=D["animals"])


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    import iti_lick_order as lo
    T, PL = F["T"], F["PL"]
    fig, axs = plt.subplots(2, 4, figsize=(19, 9.4), squeeze=False)
    x = np.arange(len(EP))
    for i, st in enumerate(SETS):
        for j, tg in enumerate(TARGETS):
            for c, val in enumerate(("r", "r_norm")):
                ax = axs[i, 2 * j + c]
                for q, m in enumerate(ROWS):
                    L = F["lines"].get((st, tg, m, val))
                    if L is None or (val == "r_norm" and m == "kinematics"):     # pre r ~0.2: a ratio is noise
                        continue
                    ax.errorbar(x + (q - 2) * 0.05, L[0], yerr=L[1], fmt="o-", color=RCOL[m], lw=1.5, ms=4.5, capsize=2,
                                label=m)
                if val == "r":
                    pl = PL[(PL.set == st) & (PL.target == tg)].set_index("model")
                    txt = [""]
                    for a_, b_ in CONTRASTS:
                        k = f"{a_} - {b_}"
                        if k in pl.index:
                            d = pl.loc[k]
                            sig = "*" if (d.lo > 0 or d.hi < 0) else ""
                            short = k.replace("cortex + behaviour", "C+B").replace("cortex", "C").replace("behaviour", "B")
                            txt.append(f"held-out pre r, {short}: {d.point:+.3f}{sig} {int(d.n_dir)}/{int(d.n)}")
                    ax.set_ylabel("decoded vs actual, r (held-out)" if j == 0 else "", fontsize=9)
                else:
                    ax.axhline(1, color="0.7", lw=0.8, ls="--")
                    for e in range(1, len(EP)):
                        for q, m in enumerate(("cortex", "behaviour", "cortex - behaviour")):
                            t = T[(T.set == st) & (T.target == tg) & (T.model == m) & (T.value == "r_norm")
                                  & (T.epoch == EP[e])]
                            if len(t):
                                col = RCOL.get(m, "0.3")
                                ax.text(e, 0.03 + 0.06 * q, ("C-B " if " - " in m else "") + nl._txt(t.iloc[0]),
                                        color=col, fontsize=6.8, ha="center", transform=ax.get_xaxis_transform())
                    ax.set_ylabel("r / row's own pre mean" if j == 0 else "", fontsize=9)
                    ax.set_ylim(0.3, max(1.3, ax.get_ylim()[1]))
                    txt = ["", "(kinematics row omitted: pre r ~0.2)"]
                ax.set_xticks(x, EP)
                ax.set_title(f"{SETS[st]} licks: {tg.upper()}, {'r' if val == 'r' else 'r / pre'}" + "\n".join(txt),
                             fontsize=8.5)
                for s_ in ("top", "right"):
                    ax.spines[s_].set_visible(False)
    h = [Line2D([], [], color=RCOL[m], marker="o", label=m) for m in ROWS]
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=8.5, frameon=False)
    cv = F["coverage"]
    cvs = "; ".join(f"{SETS[s]} " + ", ".join(f"{e} {100 * cv.loc[(s, e), 'common']:.0f}%" for e in EP if (s, e) in cv.index)
                    for s in SETS)
    fig.suptitle("Is the cortical lick-position information more than the FACE / TONGUE carry? Frozen pre-stroke ridge "
                 "decoders (KEY scheme), same licks per session, one row per feature set:\ncortex = "
                 f"{il.basis_label()} (z-scored, lags 0-0.2 s); behaviour = 5 FaceRhythm factors x 3 windows (-0.1..0.2 s, "
                 "unblanked) + 3-D reach + peak speed (no lateral / angle / landing / elevation quantity).\nMean +/- SEM "
                 "over animals. Text: change vs pre (r / pre), 95 % paired nested bootstrap, * = CI excludes 0, k/n "
                 "animals; C-B = paired contrast cortex - behaviour; 'pre ...' = held-out pre r difference.\n"
                 f"Common lick set (all features finite): {cvs}. PROVISIONAL", fontsize=9.5)
    lo._layout(fig, bottom=0.05)
    fig.subplots_adjust(top=1 - (0.17 * 5 + 0.25 + 0.55) / fig.get_size_inches()[1])   # suptitle + 3-line panel titles
    out = il.OUT_W / "figures" / f"{STEM}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--animals", nargs="*")
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args()
    il.split_only(STEM)
    if plot_only():
        plot(load_fd(STEM))
        return 0
    if "COMBINED_ENCODING_FR" not in os.environ or "COMBINED_ENCODING_FR_GROUPS" not in os.environ:
        raise SystemExit("run under the frozen env (rr2/frozenenv.sh): COMBINED_ENCODING_FR / _FR_GROUPS unset")
    D = compute(a.animals, a.jobs)
    pd.set_option("display.width", 250)
    print("== coverage (mean of sessions): common lick set, FR / kinematics finite, FR inside the frozen video mask")
    print(D["coverage"].groupby(["set", "epoch"])[["common", "fr_finite", "kin_finite", "fr_in_mask"]].mean()
          .round(3).to_string())
    print("== held-out PRE r per row and paired differences:")
    print(D["PL"].round(4).to_string(index=False))
    print("== change vs pre (per animal, rule 8) + nested bootstrap:")
    print(D["T"].round(4).to_string(index=False))
    W = il.OUT_W
    W.mkdir(parents=True, exist_ok=True)
    D["S"].to_csv(W / "behaviour_decoder_sessions.csv", index=False)
    D["T"].to_csv(W / "behaviour_decoder_tests.csv", index=False)
    D["PL"].to_csv(W / "behaviour_decoder_pre.csv", index=False)
    D["coverage"].to_csv(W / "behaviour_decoder_coverage.csv", index=False)
    F = figure_data(D)
    save_fd(STEM, F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""BEHAVIORAL RECOVERY MEASURE for ranking animals (Priya 10-27: "angle offset from intended spout as measure of recovery -
weighted for far contra then far mid, then all others") -- and THE ONE DEFINITION of the weighted signed AIM that is the
behavior axis everywhere (recovery link, mixed-model behavior model / deficit, lmm figures, aim-deficit comparison;
CLAUDE.md rule 9).

Per cued lick: angle atan2(lateral, sagittal) (degrees, 3-D, extension peak) minus the animal's PRE mean angle at the
same target spout over ALL its pre cued licks (its own pre landing, not the geometric spout direction: pre landings sit
medial of the spout, and the geometric angle carries the mouth-origin error). Per session: mean offset per spout (signed
and |.|), combined over spouts with weights far contra 3, far mid 2, every other spout 1 (renormalized over the spouts
present with >= MIN_LICKS licks). Also the far-contra offset alone (- = ipsilesional).

LICK SET (DECISIONS 2026-10-27 late, Priya): PRIMARY = the FIRST post-cue lick of each trial (lick_idx 0, joined from
first_lick_preds; the aimed movement before spout feedback; split-half reliability 0.98, r 0.94 with the all-lick
measure, chronic deficit ~2x larger). The ALL-lick measure (every cued lick) = sensitivity. ONE switch:
env WFIELD_AIM = first (default) | all. MIN_LICKS per spout and session: first 5, all 10 (the thresholds each measure
had when it was approved). The pre reference is the all-pre-lick mean in both (as approved; a constant per animal x
spout, so every change vs pre is unaffected by it).

Recovery = the session's weighted SIGNED offset; per animal x epoch means, the chronic level vs pre and the fraction of
the acute deficit remaining chronically. Pre sessions are scored the same way (their scatter is the pre noise floor).
Outputs <E>/iti_licks/recovery_measure_{sessions,animals}<AIM_TAG>.csv (AIM_TAG "" = first lick, "_aimall" = all
licks), figures/lick_cued_recovery_measure<AIM_TAG>.png, and (both lick sets, every run) the OLD vs NEW side-by-side
recovery_measure_first_vs_all_{animals,sessions}.csv (per animal first, rule 8).
Figure data <E>/iti_licks/figdata/lick_cued_recovery_measure<AIM_TAG>.pkl; --plot-only redraws from it alone.
Usage: [WFIELD_AIM=all] python scripts/iti_recovery_measure.py [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
W = {"far_contra": 3.0, "far_mid": 2.0}

# ---- THE behaviour-axis switch (DECISIONS 2026-10-27 late) ----
AIM = os.environ.get("WFIELD_AIM", "first")
if AIM not in ("first", "all"):
    raise ValueError(f"WFIELD_AIM={AIM!r}: expected first or all")
AIM_TAG = "" if AIM == "first" else "_aimall"           # primary (first lick) outputs untagged: page stems stay
MIN_LICKS = {"first": 5, "all": 10}
AIM_LABEL = {"first": "FIRST cued lick", "all": "all cued licks"}
MIN = MIN_LICKS[AIM]                                      # back-compat name


def aim_licks() -> pd.DataFrame:
    """Per cued lick: animal, label, epoch, pos, ang (deg), off (deg vs the animal's all-pre-lick mean at the spout),
    first (True = the trial's first post-cue lick, lick_idx 0)."""
    p = pd.read_csv(il.pred_file("zscore_check/where_preds_z.csv"))
    p = p[(p.set == "trial") & p.pos.isin(POS)].dropna(subset=["act_lat", "act_sag"]).copy()
    p["ang"] = np.degrees(np.arctan2(p.act_lat, p.act_sag))
    ref = p[p.epoch == "pre"].groupby(["animal", "pos"]).ang.mean()
    p["off"] = p.ang - ref.reindex(pd.MultiIndex.from_arrays([p.animal, p.pos])).to_numpy()
    # first-lick flag: join on (label, landing lateral) -- the first-lick table carries lick_idx, not a lick id
    F = pd.read_csv(il.pred_file("first_lick_preds.csv.gz"))[["label", "act", "lick_idx"]].drop_duplicates(["label", "act"])
    p["k"], F["k"] = p.act_lat.round(6), F.act.round(6)
    p = p.merge(F[["label", "k", "lick_idx"]], on=["label", "k"], how="left").drop(columns="k")
    p["first"] = p.lick_idx == 0
    return p


def weighted(g: pd.DataFrame, min_n: int, absolute: bool = False) -> float:
    """THE weighted aim of one session's licks: per spout mean offset (|offset| if absolute) over spouts with >= min_n
    licks, weights far contra 3, far mid 2, others 1, renormalized over the spouts present."""
    v = g.off.abs() if absolute else g.off
    m = v.groupby(g.pos).agg(["mean", "size"])
    m = m[m["size"] >= min_n]
    if m.empty:
        return np.nan
    w = np.array([W.get(s, 1.0) for s in m.index])
    return float((m["mean"].to_numpy() * w).sum() / w.sum())


def far_contra(g: pd.DataFrame, min_n: int, absolute: bool = False) -> float:
    h = g[g.pos == "far_contra"].off
    h = h.abs() if absolute else h
    return float(h.mean()) if len(h) >= min_n else np.nan


def session_table(p: pd.DataFrame, aim: str = AIM) -> pd.DataFrame:
    """Per session: weighted_signed (THE behavior axis), weighted_abs, far_contra_signed, far_contra_abs, n_spouts,
    n_licks, on the `aim` lick set ('first' | 'all')."""
    q = p[p["first"]] if aim == "first" else p
    mn = MIN_LICKS[aim]
    rows = []
    for (a, lab, ep), g in q.groupby(["animal", "label", "epoch"]):
        n_sp = int((g.groupby("pos").size() >= mn).sum())
        if not n_sp:
            continue
        rows.append(dict(animal=a, label=lab, epoch=ep, weighted_abs=weighted(g, mn, True),
                         weighted_signed=weighted(g, mn), far_contra_signed=far_contra(g, mn),
                         far_contra_abs=far_contra(g, mn, True), n_spouts=n_sp, n_licks=len(g), aim_licks=aim))
    from iti_check_figures import _days
    return pd.DataFrame(rows).merge(_days(), on="label", how="left")


def animal_table(S: pd.DataFrame) -> pd.DataFrame:
    A = (S.groupby(["animal", "epoch"])[["weighted_signed", "weighted_abs", "far_contra_signed", "far_contra_abs"]].mean()
         .unstack("epoch"))
    A = A.reindex(columns=EPS, level=1)
    w = A["weighted_signed"]           # PRIMARY: signed (|.| adds each animal's pre noise floor; PS93's is ~2x the others)
    A[("chronic_minus_pre_deg", "chronic")] = w["chronic"] - w["pre"]
    A[("deficit_remaining", "chronic")] = (w["chronic"] - w["pre"]) / (w["acute"] - w["pre"])
    return A


def sessions_csv(aim: str = AIM) -> Path:
    """The session table of lick set `aim` (what every behavior-axis consumer reads)."""
    return il.OUT / f"recovery_measure_sessions{'' if aim == 'first' else '_aimall'}.csv"


def aim_deficit(aim: str = AIM) -> dict:
    """{label: behavior deficit} for the mixed models' behavior model = (animal's August-pre mean - session) of the
    weighted signed aim, degrees (+ = aim shifted ipsilesional; mixed_models.behaviour_deficit, relative=False)."""
    from wfield_local import mixed_models as mx
    return mx.behaviour_deficit(sessions_csv(aim), "weighted_signed", True, relative=False)


def first_vs_all(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OLD (all licks) vs NEW (first lick) side by side: per session and per animal x epoch, with the chronic change vs
    pre and the fraction of the acute deficit remaining."""
    sf, sa = session_table(p, "first"), session_table(p, "all")
    cols = ["weighted_signed", "far_contra_signed", "weighted_abs", "n_licks"]
    S = sf[["animal", "label", "epoch", "day", *cols]].merge(sa[["label", *cols]], on="label", how="outer",
                                                             suffixes=("_first", "_all"))
    rows = []
    for a in AN:
        for nm, s in (("first", sf), ("all", sa)):
            g = s[s.animal == a].groupby("epoch").weighted_signed.mean().reindex(EPS)
            rows.append(dict(animal=a, aim_licks=nm, **{f"weighted_signed_{e}": g[e] for e in EPS},
                             chronic_minus_pre_deg=g["chronic"] - g["pre"], acute_minus_pre_deg=g["acute"] - g["pre"],
                             deficit_remaining=(g["chronic"] - g["pre"]) / (g["acute"] - g["pre"]),
                             n_sessions=int(s[s.animal == a].weighted_signed.notna().sum())))
    A = pd.DataFrame(rows)
    m = S.dropna(subset=["weighted_signed_first", "weighted_signed_all"])
    A.attrs["r_first_all"] = float(np.corrcoef(m.weighted_signed_first, m.weighted_signed_all)[0, 1])
    return S, A


def compute() -> pd.DataFrame:
    """Per-session recovery measures on the AIM lick set (+ day after stroke); writes the session / animal tables, the
    old-vs-new comparison, and prints the per-animal tables first (rule 8) and the ranking."""
    p = aim_licks()
    pd.set_option("display.width", 220)
    Sc, Ac = first_vs_all(p)
    Sc.to_csv(il.OUT / "recovery_measure_first_vs_all_sessions.csv", index=False)
    Ac.to_csv(il.OUT / "recovery_measure_first_vs_all_animals.csv", index=False)
    print("== OLD (all cued licks) vs NEW (FIRST cued lick) weighted signed aim, per animal x epoch (deg vs own pre "
          f"landing; - = ipsilesional); session r(first, all) = {Ac.attrs['r_first_all']:.2f}")
    print(Ac.round(2).to_string(index=False))
    S = session_table(p, AIM)
    S.to_csv(sessions_csv(AIM), index=False)
    A = animal_table(S)
    A.to_csv(il.OUT / f"recovery_measure_animals{AIM_TAG}.csv")
    print(f"\n== per animal (rule 8), lick set = {AIM_LABEL[AIM]}: weighted |angle offset from own pre landing| (deg), "
          "far-contra signed offset, and the fraction of the acute deficit remaining chronically")
    print(A.round(2).to_string())
    rank = A[("chronic_minus_pre_deg", "chronic")].sort_values(ascending=False)
    print("\n== recovery ranking by chronic weighted SIGNED offset vs pre (deg; closest to 0 first):",
          ", ".join(f"{a} {v:+.1f}" for a, v in rank.items()))
    return S


def plot(S: pd.DataFrame, aim: str = AIM) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import cat_points as cp
    fig, axs = plt.subplots(1, 2, figsize=(14.5, 4.6))
    for k, (val, lab) in enumerate((("weighted_signed", "weighted SIGNED angle offset from own pre (deg; - = ipsi)"),
                                    ("far_contra_signed", "far-contra signed angle offset (deg; - = ipsilesional)"))):
        ax = axs[k]
        # EPOCH-COLOUR CONVENTION (A13, Priya 11-07; was: tab10 animal colours joined by lines): open point per
        # session, shape = animal, colour = epoch, no line; filled = epoch mean +- SEM over animals
        cp.day_dots(ax, S.day, S[val], S.animal, S.epoch)
        ax.axvline(0, color="k", lw=0.7, ls="--")
        ax.axhline(0, color="0.7", lw=0.6, ls=":")
        ax.set_xlabel("day after stroke")
        ax.set_ylabel(lab, fontsize=9)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axs[1].legend(handles=cp.legend_handles(AN, [e for e in cp.EPOCH_ORDER if (S.epoch == e).any()],
                                            sessions_label="{a}", epoch_label="{e}: mean +- SEM"),
                  fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    which = ("FIRST post-cue lick of each trial (PRIMARY)" if aim == "first" else
             "ALL cued licks (sensitivity; the primary is the first lick)")
    fig.suptitle(f"BEHAVIOR: recovery measure -- {which}: angle offset from the animal's own pre landing at the target "
                 "spout,\nweighted far contra 3 : far mid 2 : others 1 (left), and far contra signed (right).\nOne dot per "
                 "session (open; shape = animal, color = epoch; filled = epoch mean +- SEM over animals); pre sessions (left of the "
                 "dashed line) = the noise floor. PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_png_svg(fig, il.OUT / "figures" / f"lick_cued_recovery_measure{'' if aim == 'first' else '_aimall'}.png", dpi=100)
    plt.close(fig)


def aim_caption(aim: str = AIM, spouts: str = "per spout") -> str:
    """THE lick set of the weighted aim (Priya 11-09): cued licks (first post-cue lick of each trial, or all), spouts
    with >= MIN_LICKS[aim] licks per session."""
    import iti_where_decoder as wd
    return wd.cued_caption(first_only=aim == "first", tracked="lateral + sagittal + outward",
                           spout="spout = the trial's target spout (one of the six)",
                           extra=[f"{spouts}: >= {MIN_LICKS[aim]} licks per session"])


def aim_short(aim: str = AIM, spouts: str = "spouts") -> str:
    """One-line form of `aim_caption` for a behavior axis drawn beside a decoder set."""
    what = "the FIRST post-cue lick of each trial (lick_idx 0)" if aim == "first" else "every post-cue lick"
    return f"{what}, {spouts} with >= {MIN_LICKS[aim]} licks per session"


def trial_set_sentence(aim: str = AIM) -> str:
    return aim_caption(aim)


def main() -> int:
    from wfield_local import trial_set_text as tst
    stem = f"lick_cued_recovery_measure{AIM_TAG}"
    if plot_only():                       # redraw from figdata only
        D = load_fd(stem)
        tst.declare(text=trial_set_sentence(D.get("aim", "all")))
        plot(D["S"], D.get("aim", "all"))
        return 0
    tst.declare(text=trial_set_sentence())
    S = compute()
    save_fd(stem, dict(S=S, aim=AIM))
    plot(S)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

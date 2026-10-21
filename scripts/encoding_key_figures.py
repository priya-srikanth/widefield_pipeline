"""Key-finding figures for the encoding / behaviour work (Priya 10-20: "are there other figures you could or should make to
illustrate key and important points or findings"). Reads tables other scripts already wrote -- no refitting.

  precue_licking.png     pre-cue (spout present, before the cue) DAQ contacts per animal x epoch, and the pre-cue
                         position dR2 (ENL window) vs the session's pre-cue lick rate -- the "ENL is not lick-free" caveat
                         (qc/precue_contacts.csv; movement_first regions)
  lick_deficits.png      acute far-contra deficits: first lick > 1.5 s (cam4 / cam4 + cam1) and no spout contact, per
                         animal x epoch and spout (qc/first_lick_latency_late_licked_trials.csv) -- slow start (PS92 / PS93)
                         vs prompt-but-miss (PS94 / PS95)
  landing_zones.png      where licks land: mean soft weight on each pre-stroke-calibrated 3-D landing zone, per target
                         spout, pre vs acute vs chronic (3-D contest sessions' res["bins"]["soft_weight"])
Output: <run>/imaging_encoding/qc/figures/. Usage: PYTHONPATH=. python scripts/encoding_key_figures.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPOCHS = ("pre", "acute", "subacute", "chronic")
ANIMAL_COL = {"PS92": "#1b9e77", "PS93": "#d95f02", "PS94": "#7570b3", "PS95": "#e7298a"}
POS = ("close_contra", "close_mid", "close_ipsi", "far_contra", "far_mid", "far_ipsi")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def precue_licking(E: Path, figs: Path) -> None:
    plt = _plt()
    P = pd.read_csv(E / "qc" / "precue_contacts.csv")
    R = pd.concat([pd.read_csv(f) for f in glob.glob(str(E / "movement_first" / "sessions" / "*_regions.csv"))])
    R = R[(R.area == "cortex") & (R.window == "enl") & R.model.isin(["full", "drop:position"])]
    dp = R.pivot_table(index="label", columns="model", values="r2")
    P["pos_enl"] = P.label.map((dp["full"] - dp["drop:position"]) * 1e3)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for an, g in P.groupby("animal"):
        m = g.groupby("epoch")[["pre_trial_frac", "pre_rate"]].mean().reindex(list(EPOCHS))
        ax[0].plot(range(4), m.pre_trial_frac, "o-", color=ANIMAL_COL[an], label=an)
        ax[1].plot(range(4), m.pre_rate, "o-", color=ANIMAL_COL[an])
        ax[2].scatter(g.pre_rate, g.pos_enl, s=14, color=ANIMAL_COL[an], alpha=0.8)
    for a_, t in zip(ax[:2], ("trials with >= 1 pre-cue spout contact", "pre-cue contacts per s (spout present)")):
        a_.set_xticks(range(4))
        a_.set_xticklabels(EPOCHS)
        a_.set_title(t, fontsize=9)
    ax[0].legend(fontsize=7)
    ax[2].set_xlabel("pre-cue contacts per s", fontsize=8)
    ax[2].set_ylabel("pre-cue position dR2 x1e3 (ENL, cortex)", fontsize=8)
    ax[2].set_title("pre-cue position code vs pre-cue licking (sessions)", fontsize=9)
    fig.suptitle("The pre-cue window is NOT lick-free: our lick tables start at the cue; the raw DAQ contact channel "
                 "shows licks before it (PS95 the exception)", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "precue_licking.png", dpi=150)
    plt.close(fig)


def lick_deficits(E: Path, figs: Path) -> None:
    plt = _plt()
    S = pd.read_csv(E / "qc" / "first_lick_latency_late_licked_trials.csv")
    cols = (("cam4_late", "first cam4 lick > 1.5 s after cue"), ("comb_late", "... with cam1 small licks added"),
            ("contact_none", "no spout contact at all"))
    sp = ("far_contra", "far_mid", "close_contra")
    fig, ax = plt.subplots(len(sp), len(cols), figsize=(12, 8), squeeze=False, sharey=True)
    for i, p in enumerate(sp):
        for j, (c, t) in enumerate(cols):
            for an, g in S[S.pos == p].groupby("animal"):
                v = g.set_index("epoch")[c].reindex(list(EPOCHS))
                ax[i, j].plot(range(4), v, "o-", color=ANIMAL_COL[an], label=an)
            ax[i, j].set_xticks(range(4))
            ax[i, j].set_xticklabels(EPOCHS, fontsize=7)
            ax[i, j].set_title(f"{p}: {t}", fontsize=8)
            ax[i, j].set_ylim(-0.03, 1.03)
    ax[0, 0].legend(fontsize=7)
    ax[0, 0].set_ylabel("fraction of licked trials", fontsize=8)
    fig.suptitle("Two acute deficits: PS92 / PS93 slow to START far-contra licking; PS94 / PS95 start promptly but "
                 "almost never reach the spout (cam1 not yet for PS94 / PS95)", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "lick_deficits.png", dpi=150)
    plt.close(fig)


def landing_zones(E: Path, figs: Path) -> None:
    plt = _plt()
    from wfield_local import epochs
    rows = []
    for f in glob.glob(str(E / "movement_first_subset-nokin_exec-polar_licks-combined_bins" / "sessions" / "*.json")):
        js = json.loads(Path(f).read_text())
        sw = (js.get("bins") or {}).get("soft_weight") or {}
        lab = js["label"]
        for t, w in sw.items():
            for z, v in w.items():
                rows.append({"label": lab, "animal": lab[:4], "epoch": epochs.epoch_of(lab), "target": t, "zone": z,
                             "w": v})
    if not rows:
        print("  landing_zones: no 3-D contest sessions with soft weights yet -- skipped")
        return
    D = pd.DataFrame(rows)
    eps = [e for e in ("pre", "acute", "chronic") if e in set(D.epoch)]
    anim = sorted(D.animal.unique())
    fig, ax = plt.subplots(len(anim), len(eps), figsize=(3.6 * len(eps), 3.3 * len(anim)), squeeze=False)
    for i, an in enumerate(anim):
        for j, ep in enumerate(eps):
            M = (D[(D.animal == an) & (D.epoch == ep)].groupby(["target", "zone"]).w.mean().unstack()
                 .reindex(index=list(POS), columns=list(POS)))
            ax[i, j].imshow(M.to_numpy(float), vmin=0, vmax=1, cmap="magma")
            for (r, c), v in np.ndenumerate(M.to_numpy(float)):
                if np.isfinite(v):
                    ax[i, j].text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6,
                                  color="w" if v < 0.5 else "k")
            ax[i, j].set_xticks(range(6))
            ax[i, j].set_xticklabels([p.replace("_", "\n") for p in POS], fontsize=6)
            ax[i, j].set_yticks(range(6))
            ax[i, j].set_yticklabels(POS, fontsize=6)
            ax[i, j].set_title(f"{an} {ep} (n={D[(D.animal == an) & (D.epoch == ep)].label.nunique()})", fontsize=8)
    fig.suptitle("Where licks land (3-D): mean soft weight on each pre-stroke landing zone (columns)\n"
                 "per aimed spout (rows); diagonal = lands in its own zone", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "landing_zones.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    a = ap.parse_args(argv)
    E = oo.root(PathResolver()) / a.run / "imaging_encoding"
    figs = E / "qc" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for f in (precue_licking, lick_deficits, landing_zones):
        try:
            f(E, figs)
            print(f"  {f.__name__}: ok")
        except FileNotFoundError as e:
            print(f"  {f.__name__}: input missing ({e})")
    print(f"-> {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

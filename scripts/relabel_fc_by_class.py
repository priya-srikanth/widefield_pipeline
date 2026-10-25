"""Far-contra TARGET trials in the frozen encoder relabel test, by ATTEMPT CLASS (Priya 10-25: "can we look at the relabel
encoder model test for far contra trials on successful vs attempted vs no attempt trials?").

Per relabel run (`frozen_relabel --tag <tag>`, trial mode: sessions/<label>.json carry `trial_sse[window][trial_id]` =
{spout label: SSE of that trial's window under the model relabelled to that spout}), far-contra trials only:
  SESSION-SUMMED (primary, more power than per-trial winners): per session x class, sum each label's SSE over the class's
  trials; % worse than the true (far contra) label = 100 * (SSE_label - SSE_fC) / SSE_fC (negative = that label fits the
  class's activity BETTER than its own target); then mean over the animal's sessions, then over animals (rule 8: per-animal
  table printed first). Best label = the label with the lowest mean % (fC itself = 0).
Classes from qc/trial_attempts.csv `cls` (hit / incomplete / jaw_only / no_try ...). The attempted-gate run (`v7g3_t3d_trial`)
has only attempted trials; the reference-gate run (`v7g3_t3d_trial_refgate`) has all engaged trials but refits its joint pre
fit on the reference gate -- compare classes WITHIN a run, not across runs.

Usage: python scripts/relabel_fc_by_class.py --tag v7g3_t3d_trial_refgate [--window cue]
Output: <imaging_encoding>/frozen_relabel_<tag>/fc_by_class_<window>{,_by_animal}.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E  # noqa: E402

SP = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")
AB = dict(zip(SP, ("cI", "cM", "cC", "fI", "fM", "fC")))
EP4 = ["pre", "acute", "subacute", "chronic"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--window", default="cue", choices=("cue", "lick"))
    ap.add_argument("--target", default="far_contra", choices=SP)
    a = ap.parse_args(argv)
    run = E / f"frozen_relabel_{a.tag}"
    att = pd.read_csv(E / "qc" / "trial_attempts.csv", usecols=["label", "trial_id", "cls"])
    rows = []
    for f in sorted(glob.glob(str(run / "sessions" / "*.json"))):
        m = json.loads(Path(f).read_text())
        for tid, v in (m.get("trial_sse") or {}).get(a.window, {}).items():
            if v.get("spout") != a.target or not all(k in v for k in SP):
                continue
            rows.append({"label": m["label"], "animal": m["animal"], "epoch": m["epoch"], "trial_id": int(tid),
                         **{k: float(v[k]) for k in SP}})
    T = pd.DataFrame(rows).merge(att, on=["label", "trial_id"], how="left")
    T["cls"] = T.cls.fillna("unclassified")
    S = T.groupby(["animal", "epoch", "cls", "label"])[list(SP)].sum()
    S["n"] = T.groupby(["animal", "epoch", "cls", "label"]).size()
    P = S[list(SP)].sub(S[a.target], axis=0).div(S[a.target], axis=0) * 100.0
    P["n"] = S.n
    A = P.groupby(["animal", "epoch", "cls"]).agg({**{k: "mean" for k in SP}, "n": "sum"})
    A["sessions"] = P.groupby(["animal", "epoch", "cls"]).size()
    A["best"] = A[list(SP)].idxmin(axis=1).map(AB)
    pd.set_option("display.width", 240)
    print(f"== {a.tag}, {a.window} window, {a.target} target trials: % SSE worse than the true label, session-summed "
          f"(negative = fits better than the target). Per animal (rule 8):")
    show = A.rename(columns=AB).reset_index()
    show["epoch"] = pd.Categorical(show.epoch, EP4, ordered=True)
    print(show.sort_values(["cls", "epoch", "animal"]).round(1).to_string(index=False))
    M = A.groupby(["epoch", "cls"])[list(SP)].mean()
    M["best"] = M.idxmin(axis=1).map(AB)
    M["n_animals"] = A.groupby(["epoch", "cls"]).size()
    M["n_trials"] = A.groupby(["epoch", "cls"]).n.sum()
    M[f"{AB[a.target]}_best_animals"] = A.assign(ok=A.best == AB[a.target]).groupby(["epoch", "cls"]).ok.sum()
    out = M.rename(columns=AB).reset_index()
    out["epoch"] = pd.Categorical(out.epoch, EP4, ordered=True)
    out = out.sort_values(["cls", "epoch"])
    print("\n== mean of animals")
    print(out.round(1).to_string(index=False))
    sfx = "" if a.target == "far_contra" else f"_{AB[a.target]}"
    out.to_csv(run / f"fc_by_class_{a.window}{sfx}.csv", index=False)
    show.to_csv(run / f"fc_by_class_{a.window}{sfx}_by_animal.csv", index=False)
    figure(out, run / "figures" / f"fc_by_class_{a.window}{sfx}.png", a)
    print("->", run)
    return 0


CLS = [("hit", "hit"), ("incomplete", "incomplete (attempt, no contact)"), ("jaw_only", "jaw only"), ("no_try", "no attempt")]


def figure(M: pd.DataFrame, path: Path, a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    labs = [AB[k] for k in SP]
    fig, axs = plt.subplots(1, len(CLS), figsize=(4.0 * len(CLS), 3.6), constrained_layout=True)
    for ax, (c, ttl) in zip(axs, CLS):
        g = M[M.cls == c].set_index("epoch").reindex(EP4)
        V = g[labs].to_numpy(float)
        im = ax.imshow(np.clip(V, -30, 30), cmap="RdBu", vmin=-30, vmax=30, aspect="auto")
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                if np.isfinite(V[i, j]):
                    ax.text(j, i, f"{V[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                            color="k" if abs(V[i, j]) < 20 else "w")
        n = g.n_trials.fillna(0).astype(int).to_numpy()
        ax.set_yticks(range(4)); ax.set_yticklabels([f"{e} (n {k})" for e, k in zip(EP4, n)], fontsize=8)
        ax.set_xticks(range(6)); ax.set_xticklabels(labs, fontsize=8)
        ax.set_title(ttl, fontsize=9.5)
    cb = fig.colorbar(im, ax=axs, shrink=0.85)
    cb.set_label(f"% SSE worse than the true label ({AB[a.target]})\nred = that label fits BETTER than the target; blue = target fits better", fontsize=8)
    fig.suptitle(f"Frozen encoder relabel, {a.target.replace('_', ' ')} trials by attempt class ({a.window} window, "
                 f"session-summed SSE, mean of animals; run {a.tag})", fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

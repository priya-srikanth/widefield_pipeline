"""Goal page Fig 9c (`fig9c_softmax_landing_stats.png` + softmax_landing_change_stats.csv): where licks land, epoch
minus pre within animal, mean of animals, nested bootstrap (animals -> sessions, 4000 draws, Bonferroni over 36 cells).

PROVENANCE: recreated 2026-11-04 (RECONCILIATION section 7) from scratchpad_scripts_backup_20261104/S1_cdd1e016/
  fig_softmax_stats.py. Computation (same seed), layout and styling unchanged (the original body, wrapped in draw()).
  Changes: inputs are THIS run's binned-contest encoder sessions (`run_mode.encoding_out_root()/<BINS>/sessions`),
  --out / --plot-only / figdata (`_common`; figdata = the session JSONs, so --plot-only re-runs the seeded bootstrap).

    PYTHONPATH=<wf_rebuild> python scripts/rebuild_figs/fig_softmax_stats.py [--out DIR] [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

BINS = "movement_first_subset-nokin_exec-polar_licks-combined_bins"


def session_jsons() -> list:
    """THIS run's binned-contest encoder sessions (`run_mode.encoding_out_root()/<BINS>/sessions/*.json`), sorted."""
    from wfield_local import run_mode
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((run_mode.encoding_out_root() / BINS / "sessions").glob("*.json"))]

NAME = "softmax_stats"


def compute() -> dict:
    return {"sessions": session_jsons()}


def draw(fd: dict, OUT: str) -> None:
    SP = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
    LAB = ["cI", "cM", "cC", "fI", "fM", "fC"]
    AN = ["PS92", "PS93", "PS94", "PS95"]; EPP = ["acute", "subacute", "chronic"]
    S = {}                                   # (animal, epoch) -> list of session 6x6 matrices
    for j in fd["sessions"]:
        sw = (j.get("bins") or {}).get("soft_weight") or {}
        M = np.full((6, 6), np.nan)
        for r, t in enumerate(SP):
            if t in sw: M[r] = [sw[t].get(e, np.nan) for e in SP]
        S.setdefault((j["animal"], j["epoch"]), []).append(M)
    rng = np.random.default_rng(0); NB = 4000
    ALPHA = 0.05 / 36                                         # Bonferroni over the 36 cells
    rows = []; res = {}
    for ep in EPP:
        ans = [a for a in AN if (a, ep) in S and (a, "pre") in S]
        obs = np.nanmean([np.nanmean(S[(a, ep)], 0) - np.nanmean(S[(a, "pre")], 0) for a in ans], 0)
        boot = np.empty((NB, 6, 6))
        for b in range(NB):
            pick = rng.choice(ans, len(ans), replace=True)
            d = []
            for a in pick:
                E_ = S[(a, ep)]; P_ = S[(a, "pre")]
                e = np.nanmean([E_[i] for i in rng.integers(0, len(E_), len(E_))], 0)
                p = np.nanmean([P_[i] for i in rng.integers(0, len(P_), len(P_))], 0)
                d.append(e - p)
            boot[b] = np.nanmean(d, 0)
        lo, hi = np.nanquantile(boot, ALPHA / 2, 0), np.nanquantile(boot, 1 - ALPHA / 2, 0)
        sig = (lo > 0) | (hi < 0)
        # per-animal sign agreement
        agree = np.sum([np.sign(np.nanmean(S[(a, ep)], 0) - np.nanmean(S[(a, "pre")], 0)) == np.sign(obs) for a in ans], 0)
        res[ep] = (obs, sig, agree, len(ans))
        for r in range(6):
            for c in range(6):
                rows.append(dict(epoch=ep, target=SP[r], executed=SP[c], mean_change=obs[r, c], lo=lo[r, c], hi=hi[r, c],
                                 significant=bool(sig[r, c]), animals_same_sign=int(agree[r, c]), n_animals=len(ans)))
    T = pd.DataFrame(rows); T.to_csv(OUT + "/softmax_landing_change_stats.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    for k, ep in enumerate(EPP):
        obs, sig, agree, n = res[ep]; ax = axs[k]
        ax.imshow(obs, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
        for r in range(6):
            for c in range(6):
                if sig[r, c]:
                    ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, ec="k", lw=2))
                if abs(obs[r, c]) >= 0.08 or sig[r, c]:
                    ax.text(c, r, f"{obs[r, c]:+.2f}\n{agree[r, c]}/{n}", ha="center", va="center", fontsize=6.5,
                            color="w" if abs(obs[r, c]) > 0.4 else "k")
        ax.set_xticks(range(6)); ax.set_xticklabels(LAB); ax.set_yticks(range(6)); ax.set_yticklabels(LAB)
        ax.set_title(f"{ep} minus pre (mean of {n} animals)"); ax.set_xlabel("executed soft bin")
        if k == 0: ax.set_ylabel("target spout")
    fig.suptitle("LICK DATA: change in where licks land, within-animal epoch minus pre, mean of animals. Boxed = nested bootstrap "
                 "(animals -> sessions, 4000 draws) excludes 0 after Bonferroni over 36 cells;\nsecond line = animals with the same "
                 "sign as the mean. Softmax width 1.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_png_svg(fig, OUT + "/fig9c_softmax_landing_stats.png", dpi=130)
    print(T[T.significant & T.target.eq("far_contra")].round(2).to_string(index=False))
    print("significant cells per epoch:", T.groupby("epoch").significant.sum().to_dict())

def trial_set_sentence() -> str:
    """The licks the soft executed-bin landing is computed from (`combined_encoding.exec_bins`)."""
    return _common.usable_licks()


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    a = ap.parse_args(argv)
    _common.declare(trial_set_sentence())          # THE TRIAL SET of every figure (Priya 11-09)
    a.out.mkdir(parents=True, exist_ok=True)
    fd = _common.load_figdata(a.out, NAME) if a.plot_only else compute()
    if not a.plot_only:
        _common.save_figdata(a.out, NAME, fd)
    draw(fd, str(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

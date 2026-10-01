"""QC of kept tongue licks that never touched the spout (no DAQ contact): real incomplete licks or false
detections? The DAQ cannot tell -- an incomplete lick makes no contact -- so this is judged on video frames.

    python -m scripts.tongue_nocontact_qc            # PS93 0908 cue clip, DLC (current lk cutoff) + LP

Priya, 2026-10-01: QC every stage visually; "DAQ won't pick up incomplete licks". Kept licks from both models
are pooled (same trial, peaks within 30 ms = one lick) so each candidate is shown once, with which model(s)
kept it. Per lick, one row: a pre-cue REST frame of the same trial (reference), frames at peak -40, -20, 0, +20,
+40 ms (zoomed on the mouth), and both models' raw tongue y (px from the mouth) over +-200 ms. DLC tongue point
green +, LP magenta x, drawn only when above the cleaning cutoff; mouth = yellow star.
The first rows are CONTACT licks (DAQ-confirmed) for calibration. Writes, next to the clip:
  nocontact_qc/page_NN.png             contact sheets
  nocontact_qc/nocontact_licks.csv     one row per candidate; `first_pass` by peak height from the mouth
                                       (extension >= 60 px, small_protrusion 45-60, tip_at_lips < 45: the
                                       tongue first becomes visible at ~30-45 px, no contact lick peaks < 53);
                                       `verdict` left blank to fill in
First result (2026-10-01, PS93 0908, DLC lk 0.4 + LP): 64 candidates, 60 kept by BOTH models; on the frames
no clear false detection -- 28 real extensions (all far positions), 15 small protrusions, 21 tip-at-lips.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import orofacial_clean as oc
from wfield_local import spout_frame as SF
from wfield_local import trial_windows as TW

MATCH_CONTACT_MS = 60.0      # a contact "is" this lick if the lick's peak lies within this of the contact onset
SAME_LICK_MS = 30.0          # two models' peaks this close = one lick
OFFSETS = (-10, -5, 0, 5, 10)  # frames around the peak (4 ms each at 250 fps)
CROP_DX, CROP_UP, CROP_DOWN = 85, 45, 175   # px around the mouth (x +-, y above / below)
REST_FRAME = 25                # clip frames after the trial's clip start (cue -0.5 s) = cue -0.4 s
TRACE_MS = 200.0
N_CONTROL = 5
PER_PAGE = 8


def kept_licks(model: str, T, clean, contacts_rel: dict) -> pd.DataFrame:
    """Every kept lick inside its trial's response window, with its clip frame and DAQ-contact match."""
    tg, pos_map, _ = clean["tongue"]
    clip_row = np.full(len(tg.y_final), -1)
    clip_row[pos_map] = np.arange(len(pos_map))
    rows = []
    for r, tr in zip(T.trial_results, T.per_trial.itertuples()):
        c = contacts_rel.get(tr.trial_id, np.zeros(0))
        for k in r.kept_licks:
            if not (70.0 <= k["t_ms"] <= tr.response_end_ms):
                continue
            f_local = int(np.argmin(np.abs(r.t_ms - k["t_ms"])))         # the peak's own frame (time is exact)
            d = np.min(np.abs(c - k["t_ms"])) if len(c) else np.inf
            rows.append({"model": model, "trial_id": tr.trial_id, "position": tr.position, "t_ms": k["t_ms"],
                         "y_from_mouth": k["y"], "x": k["x"], "source": k["source"], "imputed": k["imputed"],
                         "clip_frame": int(clip_row[r.session_frame_lo + f_local]),
                         "contact": bool(d <= MATCH_CONTACT_MS), "contact_dist_ms": float(d)})
    return pd.DataFrame(rows)


def pool(df: pd.DataFrame) -> pd.DataFrame:
    """One row per lick across models: same trial and peaks within SAME_LICK_MS."""
    out = []
    for tid, g in df.sort_values("t_ms").groupby("trial_id", sort=False):
        cur = None
        for row in g.itertuples():
            if cur is not None and row.t_ms - cur["t_last"] <= SAME_LICK_MS and row.model not in cur["models"]:
                cur["models"].append(row.model)
                cur[f"y_{row.model}"] = row.y_from_mouth
                cur["contact"] = cur["contact"] or row.contact
                cur["t_last"] = row.t_ms
                continue
            cur = {"trial_id": tid, "position": row.position, "t_ms": row.t_ms, "t_last": row.t_ms,
                   "clip_frame": row.clip_frame, "models": [row.model], f"y_{row.model}": row.y_from_mouth,
                   "contact": row.contact, "contact_dist_ms": row.contact_dist_ms}
            out.append(cur)
    p = pd.DataFrame(out)
    p["models"] = p.models.map(lambda m: "+".join(sorted(m)))
    return p.drop(columns="t_last")


def main(argv=None) -> int:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import scripts.pose_kinematics_demo as D
    from wfield_local import dlc_frames, dlc_project
    from wfield_local.paths import PathResolver

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lp", type=Path, default=Path.home() / "lp_cue_tmp" / "cue_windows_LP.csv")
    a = ap.parse_args(argv)
    rv = PathResolver()
    d = dlc_project.project_dir(rv).parent / "lp_vs_dlc_cue_traces" / "PS93_20260908"
    out = d / "nocontact_qc"
    out.mkdir(exist_ok=True)
    idx = pd.read_csv(d / "cue_windows_index.csv")
    poses = {"DLC": d / "cue_windows_DLC.csv", "LP": a.lp}
    raw = {m: oc.read_pose(f).iloc[:len(idx)] for m, f in poses.items()}
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
    frame = SF.from_medians(SF.position_medians(raw["DLC"][["spout_x", "spout_y", "spout_likelihood"]].to_numpy(), spans))
    b = TW.trial_bounds("PS93", "20260908", rv)
    stop = dict(zip(b.trial_id, b.stop_s - b.cue_s))
    sid = sorted((Path(rv.root("behavior_out")) / "sessions" / "PS93" / "20260908").glob("*_trials.csv"))[-1].name[:-11]
    licks = dlc_frames.lick_onsets("PS93", "20260908", sid, rv)
    contacts_rel = {t: (licks[(licks >= c + 0.07) & (licks <= c + s)] - c) * 1000
                    for t, c, s in zip(b.trial_id, b.cue_s, b.stop_s - b.cue_s)}
    lk_thr = oc.params("tongue")["lk_thr"]

    per_model = []
    for m, f in poses.items():
        T, _J, _M, clean = D.run(f, idx, frame, stop)
        k = kept_licks(m, T, clean, contacts_rel)
        tids = set(T.per_trial.trial_id)
        n_c = sum(len(contacts_rel.get(t, [])) for t in tids)
        found = sum(bool(len(k[k.trial_id == t])) and np.min(np.abs(k[k.trial_id == t].t_ms.to_numpy() - c)) <= MATCH_CONTACT_MS
                    for t in tids for c in contacts_rel.get(t, []))
        print(f"{m} (lk {lk_thr}): kept {len(k)}, no-contact {int((~k.contact).sum())}; contact recall {found}/{n_c}")
        per_model.append(k)
    allk = pd.concat(per_model, ignore_index=True)
    allk.to_csv(out / "kept_licks_both_models.csv", index=False)
    pooled = pool(allk)
    cand = pooled[~pooled.contact].reset_index(drop=True)
    cand.insert(0, "lick_id", [f"N{j:03d}" for j in range(len(cand))])
    ymax = cand.filter(like="y_").max(axis=1)
    cand["first_pass"] = np.select([ymax >= 60, ymax >= 45], ["extension", "small_protrusion"], "tip_at_lips")
    cand["verdict"] = ""
    cand.to_csv(out / "nocontact_licks.csv", index=False)
    print(f"pooled: {len(pooled)} licks, {len(cand)} with no DAQ contact "
          f"({cand.models.value_counts().to_dict()})")
    ctrl = pooled[pooled.contact & (pooled.models == "DLC+LP")].sample(N_CONTROL, random_state=0)
    ctrl = ctrl.assign(lick_id=[f"C{j}" for j in range(len(ctrl))])
    show = pd.concat([ctrl, cand], ignore_index=True)

    cap = cv2.VideoCapture(str(d / "cue_windows.mp4"))
    ox, oy = (int(round(v)) for v in frame.origin)
    r0, c0 = oy - CROP_UP, ox - CROP_DX
    crop = (slice(r0, oy + CROP_DOWN), slice(c0, ox + CROP_DX))
    start_of = {int(r): int(g.index.min()) for r, g in idx.groupby("trial_k")}
    trial_k_of = idx.trial_k.to_numpy()
    cols = ["rest"] + list(OFFSETS)
    pages = [show.iloc[i:i + PER_PAGE] for i in range(0, len(show), PER_PAGE)]
    for pnum, pg in enumerate(pages):
        fig, axs = plt.subplots(len(pg), len(cols) + 1, figsize=(2.1 * (len(cols) + 1.6), 3.0 * len(pg)),
                                squeeze=False, gridspec_kw={"width_ratios": [1] * len(cols) + [1.6]})
        for i, row in enumerate(pg.itertuples()):
            f0 = int(row.clip_frame)
            t0 = start_of[int(trial_k_of[f0])]
            ax = axs[i, -1]
            w = int(TRACE_MS / 4)
            ff = np.arange(max(t0, f0 - w), min(len(idx), f0 + w + 1))
            ff = ff[trial_k_of[ff] == trial_k_of[f0]]
            for m, col in (("DLC", "tab:green"), ("LP", "m")):
                pr = raw[m].iloc[ff]
                ok = pr["tongue_likelihood"].to_numpy() >= lk_thr
                yy = np.where(ok, pr["tongue_y"].to_numpy() - frame.origin[1], np.nan)
                ax.plot((ff - f0) * 4, yy, ".-", ms=2, lw=0.7, color=col, label=m)
            ax.axvline(0, color="k", lw=0.5)
            for off in OFFSETS:
                ax.axvline(off * 4, color="0.8", lw=0.4)
            ax.set_ylim(-20, 230)
            ax.invert_yaxis()
            ax.tick_params(labelsize=5)
            ax.set_title("tongue y from mouth (px; down = out)", fontsize=5)
            if i == 0:
                ax.legend(fontsize=5, loc="lower right")
            for j, off in enumerate(cols):
                ax = axs[i, j]
                ax.axis("off")
                f = t0 + REST_FRAME if off == "rest" else f0 + off
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, im = cap.read()
                if not ok:
                    continue
                ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)[crop], cmap="gray")
                ax.plot(frame.origin[0] - c0, frame.origin[1] - r0, "*", ms=7, mfc="yellow", mec="k", mew=0.5)
                for m, mk, col in (("DLC", "+", "lime"), ("LP", "x", "magenta")):
                    p = raw[m].iloc[f]
                    if p["tongue_likelihood"] >= lk_thr:
                        ax.plot(p["tongue_x"] - c0, p["tongue_y"] - r0, mk, ms=8, mew=1.5, color=col)
                if j == 0:
                    kind = "CONTACT (control)" if row.lick_id.startswith("C") else "NO CONTACT"
                    ys = " ".join(f"{m} y{getattr(row, f'y_{m}'):.0f}" for m in ("DLC", "LP")
                                  if np.isfinite(getattr(row, f"y_{m}", np.nan)))
                    ax.set_title(f"{row.lick_id} {kind}\ntrial {row.trial_id} {row.position} {row.t_ms:.0f} ms\n"
                                 f"kept by {row.models} | {ys}", fontsize=6, loc="left")
                else:
                    ax.set_title("rest (cue -0.4 s)" if off == "rest" else f"{off * 4:+d} ms", fontsize=6)
        fig.suptitle(f"PS93 0908 kept licks without spout contact, page {pnum + 1}/{len(pages)} "
                     f"(green + DLC, magenta x LP, shown if lk >= {lk_thr}; star = mouth)", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"page_{pnum + 1:02d}.png", dpi=110)
        plt.close(fig)
    cap.release()
    print(f"-> {out} ({len(pages)} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

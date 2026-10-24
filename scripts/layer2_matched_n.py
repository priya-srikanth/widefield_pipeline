"""Layer 2 combined fit with the chronic epoch cut to as many sessions as pre_late (session-count control, 10-24)."""
import json
import pickle

import numpy as np
import pandas as pd

from wfield_local import allsession_encoder as ae
from wfield_local import epochs
from wfield_local import frozen_encoder as fe

E = "M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/"
src = E + "frozen_encoder_gate-attempted_v7g3_t3d/"
pre_stats = pickle.loads(open(src + "fit/pre_stats.pkl", "rb").read())
post_stats = pickle.loads(open(E + "allsession_encoder_v7g3_t3d_ref/fit/post_stats.pkl", "rb").read())
rows = []
for an in sorted({s["animal"] for s in pre_stats}):
    S = [s for s in pre_stats + post_stats if s["animal"] == an]
    pre_s = [s for s in S if s["epoch"] == "pre"]
    col_var = pre_s[0]["col_var"]
    areas = [x for x in pre_s[0]["area_names"] if all(x in s["area_names"] for s in S)]
    sd = np.sqrt(sum(s["sxx"] for s in pre_s) / sum(s["n"] for s in pre_s))
    ok = sd > 1e-9
    scale = np.where(ok, sd, np.inf)
    sess = []
    for s in S:
        xy, y2 = fe._subset(s, areas)
        sess.append({"label": s["label"], "epoch": s["epoch"], "G": s["XtX"] / np.outer(scale, scale),
                     "b": xy / scale[:, None], "yy": y2})
    js = next(iter(sorted(__import__("pathlib").Path(src + "joint/sessions").glob(f"{an}_*.json"))))
    alphas = json.loads(js.read_text(encoding="utf-8"))["alphas"]
    groups_of = np.array([fe.FGROUP[v] for v in col_var])
    pen_of = np.array([fe.FPEN[v] for v in col_var])
    day = lambda x: epochs.days_since_stroke(x["label"])
    pre_sorted = sorted([x for x in sess if x["epoch"] == "pre"], key=day)
    half = len(pre_sorted) // 2
    late = {x["label"] for x in pre_sorted[half:]}
    n_late = len(late)
    chron = sorted([x for x in sess if x["epoch"] == "chronic"], key=day)
    for which, keep in (("first", chron[:n_late]), ("last", chron[-n_late:])):
        keep_l = {x["label"] for x in keep}
        ss = [{**x, "epoch": "pre_late"} if x["label"] in late else x for x in sess
              if x["epoch"] != "chronic" or x["label"] in keep_l]
        f = ae.fit_animal(ss, alphas, groups_of, pen_of, ok, ("pre_late", *ae.POST_EPOCHS))
        ci = areas.index("cortex")
        for x in ss:
            if x["epoch"] == "pre":
                continue
            g = (f["sse"]["ref"][x["label"]][ci] - f["sse"]["all"][x["label"]][ci]) / max(x["yy"][ci], 1e-12) * 1e3
            rows.append({"animal": an, "chronic_subset": which, "epoch": x["epoch"], "label": x["label"], "change_ref": g})
    print(an, "done", flush=True)
T = pd.DataFrame(rows)
T.to_csv(E + "allsession_encoder_v7g3_t3d_ref/matched_n_control.csv", index=False)
pd.set_option("display.width", 200)
print(T.groupby(["chronic_subset", "animal", "epoch"]).change_ref.agg(["mean", "count"]).unstack("epoch").round(1).to_string())

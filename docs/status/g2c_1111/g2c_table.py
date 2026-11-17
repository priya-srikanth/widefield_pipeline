"""Per-animal G2c table (Priya 11-11): pre-cue (plan) vs post-cue / post-lick (execution), within-session decoding vs the
animal's own pre band, from the rebuild section G JSONs (DAQ run + LP run). 8/17 (excluded) skipped. Read-only."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, r"C:/Users/SabatiniLab/Github/wf_render_v20")
FW = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/figures_working")
TAG = "hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk"
SRC = {"DAQ": FW / TAG / f"section_g__{TAG}.json", "LP": FW / f"LP_licks_gateLP_{TAG}" / f"section_g__LP_licks_gateLP_{TAG}.json"}
rows = []
for src, p in SRC.items():
    if not p.exists():
        print("MISSING", p); continue
    rec = json.loads(p.read_text(encoding="utf-8"))
    for lab, r in rec.items():
        if r.get("phase_tag") == "excluded":
            continue
        for arm, a in r.get("arms", {}).items():
            for w in ("pre-cue", "post-cue", "post-lick"):
                x = a.get(f"{w} within-session")
                if not x or not x.get("post"):
                    continue
                v = next(iter(x["post"].values())); b = x["within_pre_band"]
                rows.append(dict(src=src, arm=arm, window=w, animal=r["animal"], session=lab, acc=v["within_accuracy"],
                                 z=v["z"], inside=v["inside_pre_range"], band_mean=b["mean"], band_min=b["min"],
                                 chance=x.get("chance"), npos=x.get("n_positions")))
t = pd.DataFrame(rows)
from wfield_local import epochs
def epoch(s):
    an, d = s.split("_")
    try:
        return epochs.epoch_of(s)
    except Exception:
        return None
t["epoch"] = t.session.map(epoch)
t["below"] = (~t.inside) & (t.z < 0)
t.to_csv("g2c_sessions.csv", index=False)
t["delta"] = t.acc - t.band_mean
first = t.sort_values("session").groupby(["src", "arm", "window", "animal"]).head(1)
print("== DAY 1 (first post session): accuracy - own pre-band mean (z), * = below band")
d1 = first.assign(cell=lambda d: d.apply(lambda r: f"{r.delta:+.3f} ({r.z:+.1f}){'*' if r.below else ''}", axis=1))
print(d1.pivot_table(index=["src", "arm", "window"], columns="animal", values="cell", aggfunc="first").to_string())
print("\n== BY EPOCH: n sessions below band / n; mean delta vs own pre-band mean")
g = t.groupby(["src", "arm", "window", "animal", "epoch"]).agg(nb=("below", "sum"), n=("below", "size"), md=("delta", "mean")).reset_index()
g["cell"] = [f"{int(a)}/{b} {c:+.3f}" for a, b, c in zip(g.nb, g.n, g.md)]
print(g.pivot_table(index=["src", "arm", "window", "epoch"], columns="animal", values="cell", aggfunc="first").to_string())

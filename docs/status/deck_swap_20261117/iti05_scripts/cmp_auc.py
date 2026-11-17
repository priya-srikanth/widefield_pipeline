"""0..1.0 vs 0..0.5 s for the ITI response AUC: ramp summary (all-ITI, docked), area maps (all-ITI split), balance angle."""
import pandas as pd
from pathlib import Path

I = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks")
TAG = "hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk"
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)


def cell(t):
    return [f"{d:+.2f}{'*' if (lo > 0 or hi < 0) else ''} [{lo:+.2f},{hi:+.2f}]" for d, lo, hi in zip(t.delta, t.lo, t.hi)]


for run, new in (("all", I / "cued_ramp_iti05_20261117" / TAG), ("dock", I / "itidock_iti05_20261117" / TAG)):
    a = pd.read_csv(new / "iti_ramp_session_tests_wide.csv")
    b = pd.read_csv(new / "iti_ramp_session_tests_wide_cw.csv")
    a = a[a.value == "resp_auc_n"].assign(win="0-0.5")
    b = b.assign(win="0-1")
    t = pd.concat([b, a])
    t["c"] = cell(t)
    print(f"######## ITI response AUC / own pre, change vs pre ({run})")
    print(t.pivot_table(index="roi", columns=["epoch", "win"], values="c", aggfunc="first")
          .reindex(columns=pd.MultiIndex.from_product([["acute", "subacute", "chronic"], ["0-1", "0-0.5"]])).to_string())
    w = pd.read_csv(new / "ramp_window_tests.csv")
    w = w[(w.set == "iti") & w.value.isin(["auc_0_0.5_n", "auc_0_1_n"])]
    w["c"] = cell(w)
    print(f"-- ramp_window_tests iti ({run})")
    print(w.pivot_table(index="roi", columns=["epoch", "value"], values="c", aggfunc="first").to_string())

# area maps, all-ITI split: number of areas with CI excluding 0, resp, per epoch
old = pd.read_csv(I / TAG / "area_auc_changes_split.csv")
new = pd.read_csv(I / "area_maps_iti05_20261117" / TAG / "area_auc_changes_split.csv")
for nm, C in (("0-1 (production)", old), ("0-0.5 (new)", new)):
    c = C[C.value == "resp"]
    print(f"area maps split all-ITI {nm}: sig areas up/down per epoch",
          {ep: (int(((g.sig) & (g.delta > 0)).sum()), int(((g.sig) & (g.delta < 0)).sum())) for ep, g in c.groupby("epoch")})
    print("   chronic sig:", sorted(c[(c.epoch == "chronic") & c.sig].area))
dk = pd.read_csv(I / "area_maps_dock_iti05_20261117" / TAG / "area_auc_changes.csv")
c = dk[dk.value == "resp"]
print("area maps docked 0-0.5 (new): sig areas up/down per epoch",
      {ep: (int(((g.sig) & (g.delta > 0)).sum()), int(((g.sig) & (g.delta < 0)).sum())) for ep, g in c.groupby("epoch")})
print("   chronic sig:", sorted(c[(c.epoch == "chronic") & c.sig].area))
for f in ("area_auc_changes_split.csv", "area_auc_changes_split_farcontra.csv"):
    p = I / "area_maps_dock_iti05_20261117" / TAG / f
    if p.exists():
        c = pd.read_csv(p)
        c = c[c.value == "resp"]
        print(f"docked {f}: sig up/down", {ep: (int(((g.sig) & (g.delta > 0)).sum()), int(((g.sig) & (g.delta < 0)).sum()))
                                           for ep, g in c.groupby("epoch")})
        print("   chronic sig:", sorted(c[(c.epoch == "chronic") & c.sig].area))

# balance angle (Fig 14.13): resp measure, ITI rows
for nm, d in (("0-1 (production)", I / TAG), ("0-0.5 (new)", I / "lateral_ratio_iti05_20261117" / TAG)):
    M = pd.read_csv(d / "balance_angle_lmm.csv")
    m = M[(M.measure == "resp") & (M.set == "iti")]
    print(f"######## balance_angle_lmm {nm}, ITI resp")
    print(m.round(4).to_string(index=False))

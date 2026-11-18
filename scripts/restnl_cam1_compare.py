"""RESTnl (cam4) vs RESTnlc1 (cam4 + cam1 events_all) -- frames removed and old-vs-new tables (Priya 11-17).

Record: docs/status/RESTNL_CAM1_RERUN_20261117.md. Run under the rebuild profile (cohort1_rebuild_20261103), any block:

    python scripts/restnl_cam1_compare.py frames [--gate daq|lp]   # rest frames per animal x epoch, cam4 vs cam4 + cam1
    python scripts/restnl_cam1_compare.py pairs ROOT [ROOT ...]    # every *_restnl*.csv beside its *_restnlc1* twin

Writes CSVs to --out (default: <rebuild>/labcams/grant_figures/restnl_cam1_compare_20261117/). Reads only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

REBUILD = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103")
OUT = REBUILD / "labcams" / "grant_figures" / "restnl_cam1_compare_20261117"
EP = ["pre", "acute", "subacute", "chronic"]


def _latest(files):
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _cache_rows(d4: Path, d1: Path) -> pd.DataFrame:
    """One row per session: cam4 vs cam4cam1 baseline cache (same run tag), frames / samples / cam1 onsets."""
    from wfield_local import epochs
    from wfield_local import restnl_licks as _rl
    rows = []
    for f1 in sorted(d1.glob(f"*__{_rl.CACHE_TAG}-*__*.npz")):
        lab, rest = f1.name.split(f"__{_rl.CACHE_TAG}-", 1)
        tag = rest.rsplit("__", 1)[0]
        f4 = _latest(list(d4.glob(f"{lab}__restnl-{tag}__*.npz")))
        z1 = np.load(f1)
        i1 = json.loads(str(z1["info"]))
        r = {"label": lab, "animal": lab.split("_")[0], "epoch": epochs.epoch_of(lab), "tag": tag,
             "n_cam1_only": i1.get("n_cam1_only"), "ok_wr_c1": bool(z1["ok_wr"]),
             "frames_c1": int(np.sum(z1["nfr_pos"])), "samples_c1": i1.get("n_lickfree_samples"),
             "nfr_pos_c1": z1["nfr_pos"].tolist()}
        if f4 is not None:
            z4 = np.load(f4)
            i4 = json.loads(str(z4["info"]))
            r.update({"ok_wr_cam4": bool(z4["ok_wr"]), "frames_cam4": int(np.sum(z4["nfr_pos"])),
                      "samples_cam4": i4.get("n_lickfree_samples"), "nfr_pos_cam4": z4["nfr_pos"].tolist(),
                      "b_absdiff_med": float(np.median(np.abs(z1["b_nl"] - z4["b_nl"]))),
                      "b_rel_rms": float(np.sqrt(np.mean((z1["b_nl"] - z4["b_nl"]) ** 2))
                                         / max(np.sqrt(np.mean(z4["b_nl"] ** 2)), 1e-12))})
        rows.append(r)
    return pd.DataFrame(rows)


def frames(gate: str, out: Path) -> pd.DataFrame:
    from wfield_local import restnl_licks as _rl
    from wfield_local import session_cache
    c = Path(session_cache.CACHE_DIR)
    if gate == "daq":
        d4, d1 = c.with_name(c.name + "_restnl"), c.with_name(c.name + "_" + _rl.CACHE_TAG)
    else:
        from wfield_local import gate_source
        import os
        os.environ["WIDEFIELD_GATE_SOURCE"] = "lp"
        t = gate_source.cache_tag()
        d4, d1 = c.with_name(c.name + "_restnl_" + t), c.with_name(c.name + "_" + _rl.CACHE_TAG + "_" + t)
    S = _cache_rows(d4, d1)
    S["removed_by_cam1_pct"] = 100 * (1 - S.frames_c1 / S.frames_cam4)
    S["samples_removed_by_cam1_pct"] = 100 * (1 - S.samples_c1 / S.samples_cam4)
    out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / f"frames_by_session_{gate}gate.csv", index=False)
    g = S.groupby(["animal", "epoch"], sort=False)
    A = g.agg(sessions=("label", "size"), no_cam1=("n_cam1_only", lambda x: int((x == -1).sum())),
              cam1_only_onsets=("n_cam1_only", lambda x: int(x[x > 0].sum())),
              frames_cam4=("frames_cam4", "sum"), frames_c1=("frames_c1", "sum"),
              wr_ok_cam4=("ok_wr_cam4", "sum"), wr_ok_c1=("ok_wr_c1", "sum"),
              median_session_pct=("removed_by_cam1_pct", "median"), max_session_pct=("removed_by_cam1_pct", "max"),
              baseline_rel_rms_med=("b_rel_rms", "median")).reset_index()
    A["removed_by_cam1_pct"] = 100 * (1 - A.frames_c1 / A.frames_cam4)
    A["epoch"] = pd.Categorical(A.epoch, EP, ordered=True)
    A = A.sort_values(["animal", "epoch"])
    A.to_csv(out / f"frames_by_animal_epoch_{gate}gate.csv", index=False)
    print(f"== rest frames removed by cam1 (events_all, cam1-only, no droplets), {gate.upper()} gate, per animal x epoch")
    print(A.round(3).to_string(index=False))
    print("sessions with no cam1 events_all (cam4 only):", sorted(S.label[S.n_cam1_only == -1]))
    print("sessions where wr flips:", sorted(S.label[S.ok_wr_cam4 != S.ok_wr_c1]))
    return A


def _num(df):
    return df.select_dtypes(include=[np.number])


def pairs(roots, out: Path) -> pd.DataFrame:
    """Every `*_restnl.*csv*` (rule wr) with a `*_restnlc1.*` twin under ``roots``: shape check, max |diff| per numeric
    column, and the cells whose sign or CI-excludes-zero flag changes."""
    rows = []
    for root in roots:
        for f4 in sorted(Path(root).rglob("*_restnl.csv")) + sorted(Path(root).rglob("*_restnl.csv.gz")):
            f1 = f4.with_name(f4.name.replace("_restnl.", "_restnlc1."))
            if not f1.exists():
                continue
            try:
                a, b = pd.read_csv(f4), pd.read_csv(f1)
            except Exception as ex:                                    # noqa: BLE001
                rows.append({"file": str(f4), "error": str(ex)[:80]})
                continue
            r = {"file": str(f4.relative_to(root)), "root": str(root), "rows_old": len(a), "rows_new": len(b)}
            if a.shape == b.shape and list(a.columns) == list(b.columns):
                na, nb = _num(a), _num(b)
                d = (nb - na).abs()
                r["max_absdiff"] = float(np.nanmax(d.values)) if d.size else 0.0
                r["max_rel"] = float(np.nanmax((d / na.abs().clip(lower=1e-9)).values)) if d.size else 0.0
                sig_cols = [c for c in a.columns if re.search(r"excl|sig|star", c, re.I)]
                r["sig_changes"] = int(sum((a[c].astype(str) != b[c].astype(str)).sum() for c in sig_cols))
                mc = [c for c in na.columns if re.search(r"^(delta|mean|d|change|est|value)$", c, re.I)]
                r["sign_changes"] = int(sum((np.sign(na[c]) != np.sign(nb[c])).sum() for c in mc))
            rows.append(r)
    R = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "pairs_restnl_vs_restnlc1.csv", index=False)
    print(R.to_string(index=False) if len(R) else "no pairs found")
    return R


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("frames", "pairs"))
    ap.add_argument("roots", nargs="*")
    ap.add_argument("--gate", choices=("daq", "lp"), default="daq")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    if a.what == "frames":
        frames(a.gate, a.out)
    else:
        pairs(a.roots, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

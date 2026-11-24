"""Fig 5E side specificity under alternative TEMPLATES (CC-B 11-23, caption review item 24 + four-camera rule).

The production test (`face_rhythm_session_shape_tests.py`, `load_sessions`) scores each matched lick component's
spatial loading against the animal's pre-stroke per-spout movement maps: RMS full-point speed 0..+4 s after the cue
over the FR bundle's ENGAGED trials (behavior-pipeline engagement flag; proj ``speed_response``), averaged over ALL of
the animal's pre sessions, INCLUDING the pre session being scored. Variants (the per-session factor fits, matching and
loadings are unchanged; only the templates move):

  prod        production templates (plumbing check: the re-tested scalars must equal the stored session_scalars.csv)
  loso        production maps, leave-one-session-out for pre sessions (a pre session never scores against itself)
  pc_eng      the same maps from the 20 point PCs (`face_rhythm_aligned_fourcam.py` ``tmpl`` 0), all pre: the PC check
  pc_wp       PC maps over the FOUR-CAMERA working-period cues, all pre
  pc_wp_loso  PC maps, working period, leave-one-session-out  (candidate four-camera + LOSO definition)
  m02_wp_loso mean point speed 0..+2 s (the Fig 5B / S13 cue window), working period, leave-one-session-out

Same seed and draw order as the production run (drift control first, then the scalars), so every test but side_spec
reproduces exactly and Holm is recomputed over the same 24-test lick family.

    python scripts/face_rhythm_sidespec_templates.py [--perm 10000]
-> <readouts>/<set>/aligned_fourcam_ccB1123/sidespec/: session_scalars_<variant>.csv,
   session_scalars_per_session_<variant>.csv, sidespec_compare.csv
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import face_rhythm_cohort_null as N  # noqa: E402
import face_rhythm_session_shape_tests as st  # noqa: E402

from wfield_local.face_rhythm import common as frc  # noqa: E402

VARIANTS = ("prod", "loso", "pc_eng", "pc_wp", "pc_wp_loso", "m02_wp_loso")


def templates(maps: dict, keys_by_animal: dict, epoch_of: dict, loso: bool) -> dict:
    """key -> (6, P) template: mean of the animal's pre maps (without the key itself for a pre key when ``loso``),
    minus its mean over the six spouts."""
    out = {}
    for keys in keys_by_animal.values():
        pre = [k for k in keys if epoch_of[k] == "pre"]
        for k in keys:
            use = [q for q in pre if not (loso and q == k)]
            m = np.nanmean([maps[q] for q in use], 0)
            out[k] = m - np.nanmean(m, 0, keepdims=True)
    return out


def main(argv=None) -> int:
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import writeguard
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--perm", type=int, default=10000)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    a = ap.parse_args(argv)
    st.es.RULE = "v2"
    fdir = frc.default_set_dir(set_name=fro.SET)
    root = frc.readouts_dir(fro.SET)
    out = root / "aligned_fourcam_ccB1123" / "sidespec"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    T, names, home = st.load_sessions(fdir)
    Z = np.load(root / "aligned_fourcam_ccB1123" / "data" / "aligned_point_maps_fourcam.npz")
    zkey = list(Z["key"].astype(str))
    al = list(Z["aligns"].astype(str))
    S = frc.sessions(fdir, "proj")
    epoch_of = dict(zip(S.key, S.epoch))
    by_an = {an: list(g.key) for an, g in S.groupby("animal")}
    src = {"stored": {}, "pc_eng": {}, "pc_wp": {}, "m02_wp": {}}
    for k in S.key:
        i = zkey.index(k)
        src["stored"][k] = Z["speed_response"][i].astype(float)
        src["pc_eng"][k] = Z["tmpl"][i, 0].astype(float)
        src["pc_wp"][k] = Z["tmpl"][i, 1].astype(float)
        src["m02_wp"][k] = Z["maps"][i, al.index("cue_wp")].astype(float)
    spec = {"prod": ("stored", False), "loso": ("stored", True), "pc_eng": ("pc_eng", False),
            "pc_wp": ("pc_wp", False), "pc_wp_loso": ("pc_wp", True), "m02_wp_loso": ("m02_wp", True)}
    stored = root / "session_shape_v2" / "data" / "session_scalars.csv"
    comp = []
    for v in a.variants:
        dst = out / f"session_scalars_{v}.csv"
        if dst.exists():
            sys.exit(f"refusing to overwrite {dst}")
        s, loso = spec[v]
        tm = templates(src[s], by_an, epoch_of, loso)
        Tv = T.copy()
        own = Tv.unit.isin(list(st.OWN_SIDE))
        Tv.loc[own, "side_spec"] = [N.side_spec(ld, tm[k], st.OWN_SIDE[u])
                                    for ld, k, u in zip(Tv.load[own], Tv.key[own], Tv.unit[own])]
        rng = np.random.default_rng(0)
        st.scalars(Tv, names, a.perm, rng, drift=True)          # production draw order
        Rs = st.scalars(Tv, names, a.perm, rng)
        if v == "prod" and stored.exists() and a.perm == 10000:
            buf = io.StringIO()
            Rs.to_csv(buf, index=False, float_format="%.4g")
            same = buf.getvalue().encode().replace(b"\r\n", b"\n") == stored.read_bytes().replace(b"\r\n", b"\n")
            print("prod re-test == stored session_scalars.csv:", same, flush=True)
        Rs.to_csv(dst, index=False, float_format="%.4g")
        st.session_table(Tv, names, Rs).to_csv(out / f"session_scalars_per_session_{v}.csv", index=False)
        r = Rs[Rs.measure == "side_spec"][["epoch", "unit", "delta", "p", "p_holm"]].assign(variant=v)
        comp.append(r)
        print(v, "\n", r.round(4).to_string(index=False), flush=True)
    C = pd.concat(comp, ignore_index=True)
    C.to_csv(out / "sidespec_compare.csv", index=False, float_format="%.4g")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

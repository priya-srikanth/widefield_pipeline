"""Print the routed folder name `combined_encoding.out_dir` gives each G4 tag (read-only; no writes)."""
import json, os, sys
from pathlib import Path
from wfield_local import render_profile
render_profile.apply("cohort1_rebuild_20261123_fourcam")
B = Path(r"M:/MICROSCOPE/Priya/Widefield/rebuild_20261123_fourcam/o2/enc_fourcam3d_20261124")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import o2_encoding as oe
from wfield_local import combined_encoding as ce
from wfield_local.paths import PathResolver
import wfield_local
print("wfield_local:", wfield_local.__file__)
base = dict(os.environ)
out = {}
for t in sorted(p.name for p in (B / "out").iterdir()):
    os.environ.clear(); os.environ.update(base)
    meta = json.loads((B / "out" / t / "run.json").read_text())
    for k, v in meta["bundle_env"].items():
        if not v.startswith("@"):
            os.environ[k] = v
    oe._set_env(meta)
    if meta.get("licks_all"):
        os.environ[ce.LICKS_ALL_ENV] = meta["licks_all"]
    else:
        os.environ.pop(ce.LICKS_ALL_ENV, None)
    for k, env in (("exec", ce.EXEC_ENV),):
        if meta.get(k): os.environ[env] = meta[k]
    if meta.get("running"): os.environ["COMBINED_ENCODING_RUNNING"] = meta["running"]
    if meta.get("running_set"): os.environ["COMBINED_ENCODING_RUNNING_SET"] = meta["running_set"]
    d = ce.out_dir(PathResolver(), meta["order"] or "movement_first")
    n = len(list((B / "out" / t / "sessions").glob("PS*_*.json")))
    out[t] = d.name
    print(f"{t:22s} {n:4d} {d.name}")
print(d.parent)
json.dump(out, open(sys.argv[1], "w"), indent=1) if len(sys.argv) > 1 else None

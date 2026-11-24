"""S16 (treadmill / running supplement) re-rendered with the six-position treadmill row (S16 B) on the camera WORKING
PERIOD (trial_attempts ``canonical``, 3d_mouth classes; Priya 11-23), in the live canvas: the live figure's own builder
`manuscript_code/figures/treadmill_running/make_4x4.py` (v6 = manuscript_live/figures/S_treadmill_running.pdf), run
UNCHANGED on the live inputs except the three six-position treadmill tables, which are redirected to <wp_dir>
(treadmill_traces_by_epoch_first95.csv from `c2_treadmill_cue_aligned_first95.py` with TREADMILL_TRIAL_GATE, and the
working-period readouts / tests from `s16_treadmill_wp_ccB1123.py compare`). The other rows (two-tone treadmill, both
running rows) are the live inputs as they are. Plotting only; nothing in manuscript_code is written
(PYTHONDONTWRITEBYTECODE is forced before the import).

    MC=<manuscript_code> python s16_render_wp_ccB1123.py <live review dir> <wp_dir> <out_stem> <c1_sync60 dir>
"""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import pandas as pd  # noqa: E402

REDIRECT = ("treadmill_traces_by_epoch_first95.csv", "treadmill_session_readouts_first95.csv",
            "treadmill_tests_primary_first95.csv")


def main() -> None:
    live, wp, out, sync = (Path(a) for a in sys.argv[1:5])
    for s in (".pdf", ".png", ".svg"):
        if Path(str(out) + s).exists():
            raise SystemExit(f"refusing to overwrite {out}{s}")
    _read = pd.read_csv

    def read_csv(p, *a, **k):
        q = Path(p)
        if q.name in REDIRECT and q.parent.resolve() == live.resolve():
            print("redirect", q.name, "->", wp / q.name, flush=True)
            q = wp / q.name
        return _read(q, *a, **k)
    pd.read_csv = read_csv
    builder = Path(os.environ["MC"]) / "figures" / "treadmill_running" / "make_4x4.py"
    sys.argv = [str(builder), str(live), str(out), str(sync)]
    code = compile(builder.read_text(encoding="utf-8"), str(builder), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(builder)})


if __name__ == "__main__":
    main()

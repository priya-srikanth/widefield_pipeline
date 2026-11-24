"""Keep only camera-working-period rows of an s1 class snapshot (Fig 3D on the working period): cls_final in
contact / incomplete / jaw_only / no_response (= trial_attempts canonical under the camera gate: not stopped,
stopped_licking, unknown, no_video; isolated no-lick trials kept, as Methods). usage: in out"""
import sys

import pandas as pd
c = pd.read_csv(sys.argv[1])
w = c[c.cls_final.isin(["contact", "incomplete", "jaw_only", "no_response"])].copy()
assert (w.cls == w.cls_final).all()
print(len(c), "->", len(w), w.cls.value_counts().to_dict())
w.to_csv(sys.argv[2], index=False, float_format="%.6f")

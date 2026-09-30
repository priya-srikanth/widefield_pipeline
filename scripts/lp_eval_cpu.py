"""Evaluate the epoch-170 snapshot on CPU (GPU is busy training): labelled frames with per-frame error + split,
then the three review clips via the opencv reader."""
import sys, time, torch
torch.set_num_threads(12)
assert not torch.cuda.is_available(), "must run with CUDA_VISIBLE_DEVICES=''"
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "map_location": "cpu"})   # checkpoint was saved on GPU
from lightning_pose.api import Model
E = "/root/lp/cam4-2026-09-28/models/eval_ep170_snapshot"
m = Model.from_dir(E)
what = sys.argv[1]
t0 = time.time()
if what == "labels":
    r = m.predict_on_label_csv(E + "/CollectedData.csv", add_train_val_test_set=True)
    print("labels done", round(time.time() - t0), "s", flush=True)
else:
    for v in sys.argv[2:]:
        t0 = time.time()
        m.predict_on_video_file(v, reader="opencv", compute_metrics=False, generate_labeled_video=True)
        print("video done", v, round(time.time() - t0), "s", flush=True)

"""Tests for concat_split_session, esp. the --trim-to-sync crash-recovery path (staggered .dat/DAQ)."""
import numpy as np
import pytest

from wfield_local import concat_split_session as C


# ---- trim math (pco pulses are truth: #frames must == #pco pairs per segment) ----

def test_usable_counts_dat_longer_trims_dat():
    # .dat wrote 10 pairs but the DAQ only saw 6 pco pairs (DAQ stopped/froze first) -> keep first 6
    rs = np.arange(12) * 100 + 50
    assert C._usable_counts(10, 12, rs, 2000) == (6, 2000)   # full DAQ kept (every pulse has a frame)


def test_usable_counts_daq_longer_trims_daq():
    # .dat wrote only 4 pairs but DAQ saw 6 pco pairs (labcams froze first) -> cut DAQ after 8th edge
    rs = np.arange(12) * 100 + 50
    assert C._usable_counts(4, 12, rs, 2000) == (4, int(rs[7]) + 1)


def test_usable_counts_matched_no_trim():
    rs = np.arange(12) * 100 + 50
    assert C._usable_counts(6, 12, rs, 2000) == (6, 2000)


def test_dat_pairs_floor_tolerates_truncated_last_frame(tmp_path):
    dims = (2, 4, 4)                       # 64 B/frame-pair
    p = tmp_path / "pco_edge_run000_00000000_2_4_4_uint16.dat"
    p.write_bytes(b"\0" * (6 * 64 + 30))  # 6 whole pairs + a truncated partial
    assert C._dat_pairs_floor(p, dims) == 6
    with pytest.raises(ValueError):       # strict path refuses the partial
        C._dat_pairs(p, dims)


# ---- end-to-end concat with a staggered (trimmed) first segment ----

def _seg(tmp, name, n_pairs, pco_pairs, created, dims=(2, 4, 4), nsamp=None):
    ch, h, w = dims
    sdir = tmp / name / "raw_widefield_data"
    sdir.mkdir(parents=True)
    dat = sdir / f"pco_edge_run000_00000000_{ch}_{h}_{w}_uint16.dat"
    dat.write_bytes(np.arange(n_pairs * ch * h * w, dtype=np.uint16).tobytes())
    n_edges = 2 * pco_pairs               # invariant: pco rising edges == 2 * frame-pairs
    nsamp = nsamp if nsamp is not None else n_edges * 20 + 100
    packed = np.zeros((nsamp, 1), dtype=np.uint8)
    for k in range(n_edges):              # rising edges on bit PCO_BIT
        s = 50 + k * 20
        packed[s:s + 5, 0] |= (1 << C.PCO_BIT)
    import h5py
    h5 = tmp / f"{name}.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("analog/samples_int16", data=np.zeros((nsamp, 3), dtype=np.int16))
        f.create_dataset("analog/channel_names", data=[b"lick_analog", b"treadmill", b"reward_ttl"])
        f.create_dataset("analog/int16_scale_volts_per_count", data=np.ones(3))
        f.create_dataset("analog/int16_offset_volts", data=np.zeros(3))
        f.create_dataset("digital/packed_samples", data=packed)
        f.create_dataset("digital/channel_names", data=[b"c%d" % i for i in range(8)])
        f.attrs["sample_rate_hz"] = 5000.0
        f.attrs["created_at"] = created
        f.attrs["recording_complete"] = True
    return f"{h5}::{dat}"


def test_concat_trim_to_sync_end_to_end(tmp_path):
    import h5py
    fb = 2 * 4 * 4 * 2
    s0 = _seg(tmp_path, "PS92_20260812_100000", n_pairs=10, pco_pairs=6, created="2026-08-12T10:00:00")
    s1 = _seg(tmp_path, "PS92_20260812_100100", n_pairs=8, pco_pairs=8, created="2026-08-12T10:01:00")
    out_cam = tmp_path / "out" / "raw_widefield_data"
    out_daq = tmp_path / "out_daq.h5"
    rc = C.main(["--segment", s0, "--segment", s1, "--label", "PS92_20260812_concat",
                 "--out-cam-dir", str(out_cam), "--out-daq", str(out_daq), "--trim-to-sync"])
    assert rc == 0
    out_dat = out_cam / "pco_edge_run000_00000000_2_4_4_uint16.dat"
    assert out_dat.stat().st_size == (6 + 8) * fb          # seg0 trimmed 10->6, seg1 full 8
    with h5py.File(out_daq, "r") as h:
        assert h.attrs["concat_has_padded_gaps"]           # 60 s gap zero-padded


def test_concat_strict_refuses_mismatch_without_flag(tmp_path):
    s0 = _seg(tmp_path, "PS92_20260812_100000", n_pairs=10, pco_pairs=6, created="2026-08-12T10:00:00")
    s1 = _seg(tmp_path, "PS92_20260812_100100", n_pairs=8, pco_pairs=8, created="2026-08-12T10:01:00")
    with pytest.raises(ValueError, match="trim-to-sync"):
        C.main(["--segment", s0, "--segment", s1, "--label", "x",
                "--out-cam-dir", str(tmp_path / "o2" / "raw_widefield_data"),
                "--out-daq", str(tmp_path / "o2.h5")])

"""Tests for the config-driven preprocessing deck builder.

Exercises the pure geometry/PNG helpers and a full synthetic build, asserting that no
picture shape overflows the slide bounds (the whole point of the fit-to-box layout).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Emu

from wfield_local import preprocess_deck as pd

EPS = 0.02  # inches of slack for float rounding in overflow checks


def _make_png(path, figsize=(4, 3), dpi=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.add_subplot(111).plot([0, 1], [0, 1])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# fit_dims
# ---------------------------------------------------------------------------
def test_fit_dims_wide_is_width_limited():
    box_w, box_h = 12.9, 6.4
    w, h = pd.fit_dims(1900, 550, box_w, box_h)
    assert abs(w - box_w) < 1e-9          # width-limited: fills the box width
    assert h <= box_h + 1e-9
    assert w <= box_w + 1e-9
    assert abs((w / h) - (1900 / 550)) < 1e-6  # aspect preserved


def test_fit_dims_tall_is_height_limited():
    box_w, box_h = 12.9, 6.4
    w, h = pd.fit_dims(800, 1600, box_w, box_h)
    assert abs(h - box_h) < 1e-9          # height-limited
    assert w <= box_w + 1e-9
    assert h <= box_h + 1e-9
    assert abs((w / h) - (800 / 1600)) < 1e-6


def test_fit_dims_never_exceeds_box():
    box_w, box_h = 12.9, 6.4
    for px in [(1900, 550), (800, 1600), (640, 540), (1000, 1000), (2500, 300)]:
        w, h = pd.fit_dims(*px, box_w, box_h)
        assert w <= box_w + 1e-9 and h <= box_h + 1e-9


# ---------------------------------------------------------------------------
# _png_size
# ---------------------------------------------------------------------------
def test_png_size_reads_ihdr(tmp_path):
    p = tmp_path / "known.png"
    _make_png(p, figsize=(4, 3), dpi=100)   # -> 400 x 300 px
    w, h = pd._png_size(str(p))
    assert (w, h) == (400, 300)


def test_png_size_rejects_non_png(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"not a png")
    try:
        pd._png_size(str(p))
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# build smoke test on a synthetic tree
# ---------------------------------------------------------------------------
def _synthetic_sessions(tmp_path):
    """Two PS92 sessions with a subset of real figure files, plus labcams/xday roots."""
    labcams = tmp_path / "labcams"
    xday = tmp_path / "xday"
    sessions = []
    specs = [
        ("0607", "20260607", "PS92_20260607_121538"),
        ("0608", "20260608", "PS92_20260608_133759"),
    ]
    for mmdd, ymd, sess in specs:
        mc = labcams / ymd / sess / "motion_corrected"
        lab = f"PS92_{mmdd}_affine8v1"
        # spout_trial_averages
        st = mc / "spout_trial_averages_affine8v1"
        _make_png(st / f"{lab}_mean_415_470_with_allen_overlay.png", (6, 3))
        _make_png(st / f"{lab}_spout_positions_1s_pre_post_delta_shared_scale.png", (3, 6))
        # lick_aligned
        la = mc / "lick_aligned_affine8v1"
        _make_png(la / f"{lab}_lick_aligned_150ms_post_by_spout.png", (8, 2))
        _make_png(la / f"{lab}_cue_vs_lick_spout_position_maps.png", (4, 4))
        # motion_qc
        _make_png(mc / "motion_qc" / f"{lab}_motion_qc.png", (7, 3))
        sessions.append(dict(label=f"PS92_{mmdd}", mc=str(mc).replace("\\", "/"),
                             h5="", regime=None, fmdir=None))
    # per-animal cross-day QC under <xday>/PS92_xall/
    _make_png(xday / "PS92_xall" / "PS92_cross_day_alignment_qc.png", (6, 4))
    return sessions, str(labcams), str(xday)


def test_build_deck_smoke_no_overflow(tmp_path):
    sessions, labcams, xday = _synthetic_sessions(tmp_path)
    out = tmp_path / "deck.pptx"
    summary = pd.build_deck(str(out), sessions=sessions,
                            labcams_root=labcams, xday_root=xday, verbose=False)
    assert out.exists()
    assert summary["total_slides"] > 0
    assert summary["per_animal"].get("PS92", 0) > 0
    # types with no synthetic figures should report zero
    assert "cue_pairwise" in summary["zero_types"]
    assert "photobleach" in summary["zero_types"]

    prs = Presentation(str(out))
    sw = Emu(prs.slide_width).inches
    sh = Emu(prs.slide_height).inches
    n_pics = 0
    for slide in prs.slides:
        for shp in slide.shapes:
            if shp.shape_type == 13:  # PICTURE
                n_pics += 1
                left = Emu(shp.left).inches
                top = Emu(shp.top).inches
                w = Emu(shp.width).inches
                h = Emu(shp.height).inches
                assert left + w <= sw + EPS, (left, w, sw)
                assert top + h <= sh + EPS, (top, h, sh)
                assert left >= -EPS and top >= -EPS
    assert n_pics > 0


def test_build_deck_date_ascending_and_animal_divider(tmp_path):
    sessions, labcams, xday = _synthetic_sessions(tmp_path)
    out = tmp_path / "deck2.pptx"
    pd.build_deck(str(out), sessions=sessions,
                  labcams_root=labcams, xday_root=xday, verbose=False)
    prs = Presentation(str(out))

    def title_of(slide):
        for shp in slide.shapes:
            if shp.has_text_frame and shp.text_frame.text.strip():
                return shp.text_frame.text.strip()
        return ""

    titles = [title_of(s) for s in prs.slides]
    assert titles[0] == "PS92"  # animal divider first
    # allen slides appear grouped, 0607 before 0608 (date ascending)
    allen = [t for t in titles if "Allen alignment" in t]
    assert len(allen) == 2 and "2026-06-07" in allen[0] and "2026-06-08" in allen[1]


# ---------------------------------------------------------------------------
# size split: partition_animals + build_decks
# ---------------------------------------------------------------------------
def _sessions_with_counts(counts):
    out = []
    for a, n in counts.items():
        for i in range(n):
            out.append(dict(label=f"{a}_{i:02d}", mc=f"/x/2026{i:02d}01/{a}/motion_corrected"))
    return out


def test_partition_animals_packs_whole_animals():
    order = ["PS92", "PS93", "PS94", "PS95"]
    S = _sessions_with_counts({"PS92": 10, "PS93": 7, "PS94": 10, "PS95": 10})
    # cap 10 -> one deck per animal (PS92 fills exactly; PS93 can't join PS94, etc.)
    assert [a for a, _ in pd.partition_animals(S, order, 10)] == [["PS92"], ["PS93"], ["PS94"], ["PS95"]]
    # cap 20 -> two decks, whole animals, never exceeding the cap
    b20 = pd.partition_animals(S, order, 20)
    assert [a for a, _ in b20] == [["PS92", "PS93"], ["PS94", "PS95"]]
    assert [n for _, n in b20] == [17, 20]
    # cap 0/None -> single bucket with all animals
    assert pd.partition_animals(S, order, 0) == [(order, 37)]


def test_partition_animal_over_cap_gets_own_bucket():
    order = ["PS92", "PS93"]
    S = _sessions_with_counts({"PS92": 15, "PS93": 3})   # PS92 alone exceeds cap 10
    assert pd.partition_animals(S, order, 10) == [(["PS92"], 15), (["PS93"], 3)]


def _two_animal_tree(tmp_path):
    labcams, xday, sessions = tmp_path / "labcams", tmp_path / "xday", []
    for animal in ("PS92", "PS93"):
        for mmdd in ("0607", "0608"):
            ymd = f"2026{mmdd}"
            mc = labcams / ymd / f"{animal}_{ymd}_x" / "motion_corrected"
            lab = f"{animal}_{mmdd}_affine8v1"
            _make_png(mc / "spout_trial_averages_affine8v1"
                      / f"{lab}_mean_415_470_with_allen_overlay.png", (6, 3))
            sessions.append(dict(label=f"{animal}_{mmdd}", mc=str(mc).replace("\\", "/"),
                                 h5="", regime=None, fmdir=None))
    return sessions, str(labcams), str(xday)


def test_build_decks_splits_by_animal_and_prunes_stale(tmp_path, monkeypatch):
    sessions, labcams, xday = _two_animal_tree(tmp_path)
    # Pruning is only permitted when the run covers EVERY configured animal (a partial run must not
    # delete another machine's decks -- see tests/test_shared_output_safety.py). This tree holds the
    # whole cohort, so declare it as such and the stale-sibling prune stays enabled.
    monkeypatch.setattr(pd.config, "animals", lambda: {"PS92": {}, "PS93": {}})
    base = tmp_path / "deck.pptx"
    (tmp_path / "deck_PS94.pptx").write_bytes(b"stale")   # leftover from an earlier split
    summ = pd.build_decks(str(base), sessions=sessions, max_sessions=2,
                          labcams_root=labcams, xday_root=xday, verbose=False)
    assert len(summ) == 2
    assert (tmp_path / "deck_PS92.pptx").exists() and (tmp_path / "deck_PS93.pptx").exists()
    assert not base.exists()                              # split -> no plain base file
    assert not (tmp_path / "deck_PS94.pptx").exists()     # stale sibling pruned


def test_build_decks_single_file_when_cap_zero(tmp_path):
    sessions, labcams, xday = _two_animal_tree(tmp_path)
    base = tmp_path / "deck.pptx"
    summ = pd.build_decks(str(base), sessions=sessions, max_sessions=0,
                          labcams_root=labcams, xday_root=xday, verbose=False)
    assert len(summ) == 1 and base.exists()
    assert not (tmp_path / "deck_PS92.pptx").exists()     # not split

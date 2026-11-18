"""Deck images for the quiet-event slides from the sync-end run (Priya 11-18: "yes you can swap the figures and numbers").
Run v2_quiet_lp_c1guard_syncend_20261118 (QS_ENGAGED_END=lastsync, k 4); slides 157-159 use the restyled figures
(restyled_20261118/, quiet_restyle_figs.py: bootstrap * primary, mixed-model dagger, change vs own pre, session markers),
whose layouts differ from the old renders, so their cutters are new pixel boxes on the restyled canvases. 156 / 204 reuse
the c1guard_crops.py cutters (same render size as before). Slide 155 is a movie (no image).
python syncend_crops.py make   -> cut into swap_inputs/ (refuses to overwrite)"""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1guard_crops import SWAP, Q, eta, examples, trim  # noqa: E402

RUN = Q / "v2_quiet_lp_c1guard_syncend_20261118"
RS = RUN / "restyled_20261118"


def stack(parts, gap=6):
    W = max(p.width for p in parts)
    c = Image.new("RGB", (W, sum(p.height for p in parts) + gap * (len(parts) - 1)), "white")
    y = 0
    for p in parts:
        c.paste(p, ((W - p.width) // 2, y))
        y += p.height + gap
    return trim(c)


def fr_2x2(p):             # legend + 4 panels; suptitle and footer dropped (the slide caption carries them)
    im = Image.open(p).convert("RGB")
    return trim(im.crop((0, 35, im.width, 785)))


def stillness(p):          # legend over the left (event rate by face stillness) panel
    im = Image.open(p).convert("RGB")
    return stack([trim(im.crop((370, 32, 1280, 58))), trim(im.crop((0, 95, 818, 548)))])


def hemo(p):               # legend, corrected-470 row (with column titles), 415 row; all five columns
    im = Image.open(p).convert("RGB")
    return stack([trim(im.crop((480, 52, 1400, 82))), trim(im.crop((0, 118, im.width, 434))),
                  trim(im.crop((0, 712, im.width, 990)))])


def t13(p):                # lower-left panel: frozen decoder excess p_true, session level
    im = Image.open(p).convert("RGB")
    return trim(im.crop((0, 598, 596, 1078)))


JOBS = {156: (RUN / "quiet_event_example_traces.png", examples, "s156_quiet_examples_PS92_syncend.png"),
        157: (RS / "quiet_events_face_regressed_2x2_restyled.png", fr_2x2, "s157_quiet_fr_2x2_syncend.png"),
        "158a": (RS / "quiet_stillness_restyled.png", stillness, "s158_q_stillness_rate_syncend.png"),
        "158b": (RS / "quiet_hemo_check_restyled.png", hemo, "s158_q_hemo415_rows_syncend.png"),
        159: (RS / "t13_precue_event_vs_outcome_restyled.png", t13, "s159_t13_decoder_syncend.png"),
        204: (RUN / "eta_fr" / "quiet_eta_maps_fr.png", eta, "s204_quiet_eta_maps_orofacial_syncend.png")}

if __name__ == "__main__":
    assert sys.argv[1:] == ["make"], __doc__
    for k, (src, cut, name) in JOBS.items():
        out = SWAP / name
        if out.exists():
            raise SystemExit(f"exists, not overwriting: {out}")
        im = cut(src)
        im.save(out)
        print(k, im.size, out)

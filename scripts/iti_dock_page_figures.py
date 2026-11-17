"""Collect the DOCKED-SPOUT versions of the main-page ITI figures (Priya 10-29 late: "replace the main ITI figures with
this (if they support the same conclusions)") under NEW names <stem>_docked.png.

Each source PNG was drawn by its own script under WFIELD_ITI_FILTER=dock (frozen all-ITI decoders, ITI rows subset to
licks whose onset falls between the spout reaching the dock and the next trial_start; iti_dock_audit.py) into
<OUT>/itidock_main/ (WFIELD_ITI_OUT_W), or, for Fig 14.18, the earlier helper's <OUT>/itifilter_dock/. A one-line banner
is added ABOVE the figure (canvas extended, nothing drawn over the data) so a docked image can never be mistaken for the
all-ITI one. Writes <OUT>/itidock_main/figures_docked/<stem>_docked.png and copies them to --page DIR (default: none).
The all-ITI originals are never touched.
Usage: python scripts/iti_dock_page_figures.py [--page _goal_page]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

DOCK = il.OUT / "itidock_main"
OLD = il.OUT / "itifilter_dock"
# page figure -> source folder (stems are the page image names)
FIGS = {
    "10.2": [(DOCK, "lick_where_decoder_z")],
    "12.1": [(DOCK, "lick_matched_gauss_readout")],
    "12.2": [(DOCK, "lick_matched_gauss_curves"), (DOCK, "lick_matched_gauss_nonlinear_curves")],
    "12.3": [(DOCK, "lick_iti_check_error_per_lick")],
    "12.6": [(DOCK, "lick_iti_where_scatter_z")],
    "12.7": [(DOCK, "lick_where_delta_z")],
    "12.8": [(DOCK, "lick_iti_check_error_sessions")],
    "12.15": [(DOCK, "lick_sametarget_place")],
    "12.19": [(DOCK, "lick_balance_offset_primary")],
    "12.21": [(DOCK, "balblind_offset"), (DOCK, "balblind_subsets")],
    "13.1": [(DOCK, "lick_order_iti_cued")],
    "13.2": [(DOCK, "lick_order_iti_cued_slope"), (DOCK, "lick_order_iti_cued_slope_nonlinear")],
    "13.3": [(DOCK, "lick_where_per_spout_z")],
    "13.6": [(DOCK, "lick_matched_gauss_perspout")],
    "13.7": [(DOCK, "lick_position_per_spout_byepoch"), (DOCK, "lick_position_per_spout")],
    "13.8": [(DOCK, "lick_decoded_decomposition")],
    "14.1": [(DOCK, "lick_iti_ramp_wide")],
    "14.2": [(DOCK, "lick_iti_ramp_animals")],
    "14.7": [(DOCK, "lick_iti_ramp_perspout")],
    "14.9": [(DOCK, "lick_lateral_ratio_auc"), (DOCK, "lick_lateral_ratio_auc_0-0.5"),
             (DOCK, "lick_lateral_ratio_traces_iti")],
    "14.10": [(DOCK, "lick_lateral_ratio_side_auc")],
    "14.14": [(DOCK, "lick_balance_order_primary")],
    "14.18": [(OLD, "lick_recovery_primary")],
    "14.22": [(DOCK, "lick_iti_ramp_summary_wide")],
    "14.28": [(DOCK, "lick_iti_area_maps_split"), (DOCK, "lick_iti_area_maps_split_farcontra"),
              (DOCK, "lick_iti_area_maps_split_levels"), (DOCK, "lick_iti_area_maps_split_farcontra_levels")],
    "14.29": [(DOCK, "lick_iti_area_peth")],
    "14.30": [(DOCK, "lick_iti_area_compare")],
    "14.31": [(DOCK, "lick_iti_area_groups")],
    "16.1": [(DOCK, "lmm_time_readout"), (DOCK, "lmm_epoch_readout")],
    "16.3": [(DOCK, "lick_recovery_link_pooled_aim_z")],
    "16.4": [(DOCK, "lick_recovery_link_aim_z")],
    "16.5": [(DOCK, "lick_recovery_link_sessions_aim_z")],
    "16.7": [(DOCK, "lmm_time_lickclass")],
    "16.8": [(DOCK, "lmm_time_perspout")],
}
BANNER = ("DOCKED-SPOUT ITI LICKS (onset between the spout reaching the dock and the next trial start; "
          "frozen all-ITI decoders, nothing refit) -- cued rows unchanged")


def banner(src: Path, dst: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(src).convert("RGB")
    w, h = im.size
    size = max(14, min(22, w // 75))
    try:
        font = ImageFont.truetype("arialbd.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    pad = int(size * 0.6)
    bh = size + 2 * pad
    out = Image.new("RGB", (w, h + bh), "white")
    out.paste(im, (0, bh))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, w, bh - 2], fill=(27, 158, 119))
    tw = d.textlength(BANNER, font=font)
    d.text((max(pad, (w - tw) / 2), pad), BANNER, fill="white", font=font)
    out.save(dst)


def main() -> int:
    page = Path(sys.argv[sys.argv.index("--page") + 1]) if "--page" in sys.argv else None
    od = DOCK / "figures_docked"
    od.mkdir(parents=True, exist_ok=True)
    missing = []
    for fig, items in FIGS.items():
        for root, stem in items:
            src = root / "figures" / f"{stem}.png"
            if not src.exists():
                missing.append(f"{fig}: {src}")
                continue
            dst = od / f"{stem}_docked.png"
            banner(src, dst)
            if page:
                shutil.copy2(dst, page / dst.name)
            print(f"Fig {fig}: {dst.name}")
    if missing:
        print("MISSING:\n  " + "\n  ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

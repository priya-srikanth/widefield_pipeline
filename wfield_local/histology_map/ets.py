"""Crop regions from an Olympus VS200 .vsi/.ets pyramid at a chosen level, all channels, no Bio-Formats.

ETS layout: SIS header -> ETS header (pixel type, sizeC, compression, tile size) -> chunk table of
(x, y, c, level) -> (offset, size). VS200 10x fluorescence scans here are JPEG2000 (compression 3) 512x512 uint16
tiles, one channel per tile; Pillow decodes them. The 10x image of ``<name>.vsi`` is the largest ``.ets`` under
``_<name>_/`` (e.g. ``_01.vsi`` -> ``__01_/stack10002/frame_t_0.ets``). Coordinates match QuPath's full-resolution
pixels (verified on PS92-95: the QuPath outlines land on the striata).

Run where the scans are (it only needs numpy + Pillow, both on the Mac's Cowork VM)::

    python -m wfield_local.histology_map.ets jobs.json OUTDIR 3 [PS92_s1 ...]

``jobs.json`` = list of {animal, section, uri, cx, cy}; writes ``OUTDIR/<animal>_s<n>_L<level>.npz`` with
``img`` (C, H, W uint16), ``origin`` (level-0 x, y of the crop), ``ds`` (2**level).
"""
from __future__ import annotations

import io
import json
import os
import struct
import sys
import time

import numpy as np
from PIL import Image

HALF_W, UP, DOWN = 7500, 5600, 4600     # crop box around a section centre, level-0 px (~9.8 x 6.6 mm)


def ets_path(uri: str) -> str:
    vsi = uri.replace("file:", "")
    d, stem = os.path.dirname(vsi), os.path.splitext(os.path.basename(vsi))[0]
    best = None
    for root, _, files in os.walk(os.path.join(d, "_" + stem + "_")):
        for f in files:
            if f.endswith(".ets"):
                p = os.path.join(root, f); s = os.path.getsize(p)
                if best is None or s > best[1]:
                    best = (p, s)
    if best is None:
        raise FileNotFoundError(f"no .ets under _{stem}_ for {vsi}")
    return best[0]


class ETS:
    def __init__(self, path):
        self.f = open(path, "rb")
        h = self.f.read(64)
        _, _, _, ndim, addoff, addsz, _, usedoff, nused, _ = struct.unpack("<4siiiqiiqii", h[:48])
        self.f.seek(addoff); e = self.f.read(addsz)
        v = struct.unpack("<" + "i" * 9, e[4:40])
        self.sizeC, self.comp, self.tile = v[2], v[4], v[6]
        self.f.seek(usedoff); rec = 4 + 4 * ndim + 16; tab = self.f.read(rec * nused)
        self.idx = {}
        for i in range(nused):
            r = tab[i * rec:(i + 1) * rec]
            c = struct.unpack("<" + "i" * ndim, r[4:4 + 4 * ndim])
            off, size = struct.unpack("<qi", r[4 + 4 * ndim:16 + 4 * ndim])
            self.idx[(c[0], c[1], c[2], c[-1])] = (off, size)
        self.nc = 1 + max(k[2] for k in self.idx)

    def level_shape(self, L):
        xs = [k[0] for k in self.idx if k[3] == L]; ys = [k[1] for k in self.idx if k[3] == L]
        return (max(ys) + 1) * self.tile, (max(xs) + 1) * self.tile

    def tile_img(self, x, y, c, L):
        k = (x, y, c, L)
        if k not in self.idx:
            return np.zeros((self.tile, self.tile), np.uint16)
        off, size = self.idx[k]; self.f.seek(off)
        return np.array(Image.open(io.BytesIO(self.f.read(size))))

    def crop(self, x0, y0, x1, y1, L):
        """Crop in LEVEL-0 pixel coords; returns ((C, H, W) at level L, level-0 origin of the crop)."""
        ds = 2 ** L; t = self.tile
        X0, Y0, X1, Y1 = int(max(x0, 0) // ds), int(max(y0, 0) // ds), int(x1 // ds), int(y1 // ds)
        out = np.zeros((self.nc, Y1 - Y0, X1 - X0), np.uint16)
        for ty in range(Y0 // t, (Y1 - 1) // t + 1):
            for tx in range(X0 // t, (X1 - 1) // t + 1):
                for c in range(self.nc):
                    a = self.tile_img(tx, ty, c, L); ya, xa = ty * t, tx * t
                    sy0, sx0 = max(Y0, ya), max(X0, xa); sy1, sx1 = min(Y1, ya + t), min(X1, xa + t)
                    if sy1 > sy0 and sx1 > sx0:
                        out[c, sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = a[sy0 - ya:sy1 - ya, sx0 - xa:sx1 - xa]
        return out, (X0 * ds, Y0 * ds)


def crop_jobs(jobs, outdir, level=3, only=()):
    os.makedirs(outdir, exist_ok=True)
    for j in jobs:
        name = f"{j['animal']}_s{j['section']}"
        if only and name not in only:
            continue
        out = os.path.join(outdir, f"{name}_L{level}.npz")
        if os.path.exists(out):
            print("skip", name); continue
        t0 = time.time(); p = ets_path(j["uri"]); e = ETS(p)
        img, org = e.crop(j["cx"] - HALF_W, j["cy"] - UP, j["cx"] + HALF_W, j["cy"] + DOWN, level)
        np.savez_compressed(out, img=img, origin=np.array(org), level=level, ds=2 ** level, ets=p,
                            level0_shape=np.array(e.level_shape(0)))
        print(name, img.shape, round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    crop_jobs(json.load(open(sys.argv[1])), sys.argv[2], int(sys.argv[3]), tuple(sys.argv[4:]))

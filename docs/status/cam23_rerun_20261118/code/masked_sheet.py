"""Contact sheet: masked cam1 runs that overlap a cam4 lick vs masked runs that do not (scratch, 11-18)."""
import sys
import cv2
import numpy as np, pandas as pd
sys.path.insert(0, "C:/Users/SabatiniLab/Github/wf_docbranches")
S = sys.argv[1]


def strips(s, rows):
    from scripts.cam1_licks import PROD, _video, OFFSETS_MS
    from wfield_local import cam1_licks as C1, o2_inference as oi, spout_frame as SF
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    a, d = s.split("_")
    c1 = oi.session_windows(rv, PROD["cam1"], a, d, "cam1")
    c4 = oi.session_windows(rv, PROD["cam4"], a, d, "cam4")
    ix1, fl1 = c1[0].reset_index(drop=True), c1[1].reset_index(drop=True)
    fr1 = C1.cam1_frame(fl1, ix1)[0]
    o4 = SF.from_medians(SF.ruler_medians(c4[1], c4[0])).origin
    out = []
    for r in rows.itertuples():
        t0 = (r.t0 + r.t1) / 2
        st = []
        for cam, ix, fl, origin, shift, crop in (("cam1", ix1, fl1, fr1.origin, -10.0 * np.asarray(fr1.ap_axis), (260, 220)),
                                                 ("cam4", c4[0], None, o4, np.array([0.0, 40.0]), (300, 240))):
            cap = cv2.VideoCapture(str(_video(rv, a, d, cam, ix, r.trial_id)))
            h = ix[ix.trial_id == r.trial_id]
            tiles = []
            for off in OFFSETS_MS:
                i = int(np.argmin(np.abs(h.t_ms.to_numpy(float) - (t0 + off))))
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(h.src_frame.iloc[i]))
                ok, im = cap.read()
                if not ok:
                    im = np.zeros((1024, 1280, 3), np.uint8)
                if fl is not None:
                    q = fl.iloc[h.index[i]]
                    if q.tongue_likelihood >= 0.4:
                        cv2.circle(im, (int(q.tongue_x), int(q.tongue_y)), 9, (255, 0, 255), 1, cv2.LINE_AA)
                cx, cy = (np.asarray(origin, float) + shift).astype(int)
                x0, y0 = max(cx - crop[0] // 2, 0), max(cy - crop[1] // 2, 0)
                tl = cv2.resize(im[y0:y0 + crop[1], x0:x0 + crop[0]], (150, int(150 * crop[1] / crop[0])))
                cv2.putText(tl, f"{off:+d}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tl)
            cap.release()
            st.append(np.hstack(tiles))
        img = np.vstack(st)
        lab = f"{s} t{r.trial_id} {r.t0:.0f}..{r.t1:.0f}ms n{r.n} out{r.min_out:.0f}-{r.max_out:.0f} sp{r.spread:.1f} {'IN-cam4' if r.cam4 else 'no-cam4'}"
        bar = np.zeros((18, img.shape[1], 3), np.uint8)
        cv2.putText(bar, lab, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        out.append((r.Index, np.vstack([bar, img])))
    return out


if __name__ == "__main__":
    R = pd.read_csv(f"{S}/runs.csv.gz")
    B = R[R.blast & (R.min_out > 30) & (R.spread < 6)]
    rng = np.random.default_rng(1)
    pick = pd.concat([B[B.cam4].sample(12, random_state=1), B[~B.cam4 & (B.n >= 3)].sample(6, random_state=2)])
    res = []
    for s, g in pick.groupby("session"):
        res += strips(s, g)
    order = {i: k for k, i in enumerate(pick.index)}
    res.sort(key=lambda x: order[x[0]])
    ims = [im for _i, im in res]
    for k in range(0, len(ims), 6):
        cv2.imwrite(f"{S}/masked_sheet_{k // 6}.png", np.vstack(ims[k:k + 6]))
    pick.to_csv(f"{S}/masked_sheet_key.csv")
    print("sheets", (len(ims) + 5) // 6)

balance_aim_sessions.csv: cued first lick, from
rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/<daqlk tag>/figdata/lick_recovery_primary.pkl  (T, set == 'cued first lick')
marker = angle-matched gain ratio, log2(obs/pred contra) - log2(obs/pred ipsi), pool ALM+tjM1+SSp-m+SSp-n+SSs,
0.1-0.5 s after the first cued lick; pred = animal's PRE line on lick angle. y = far-contra first-lick aim (deg vs pre).
Stats from M (same pkl) and balimg_primary_link.csv: concurrent MixedLM y ~ w + b + log(day), slope 13.6 [6.2, 21.0], p 3.3e-4, n 76 post sessions;
per-animal spline 17.5, p 5.1e-5; predictive (next-session change) 0.5, p 0.91; per-animal r 0.69 / 0.89 / 0.93 / 0.75.

"""False-positive rate + power at n = 4 animals: animal-weighted bootstrap (analysis_kit.paired_animal_draws, the
A3 code, 95% percentile interval excludes 0) vs one-sample t on the 4 animal changes (df 3).
Session counts = the acute hit-rate cells (pre 16/12/16/16, post 5/4/6/1). Animal changes ~ N(effect, tau);
session noise sd sigma. Null: effect = 0."""
import sys
import numpy as np
from scipy import stats
sys.path.insert(0, ".")
from wfield_local import analysis_kit as ak
NPRE, NPOST = [16, 12, 16, 16], [5, 4, 6, 1]
mean_of = lambda v, _r=None: float(np.mean(v))
def sim(effect, tau, sigma, n_sim, seed):
    rng = np.random.default_rng(seed); fb = ft = 0
    for _ in range(n_sim):
        ch = rng.normal(effect, tau, 4); base = rng.normal(0, 0.05, 4)
        pre = {f"A{i}": list(base[i] + rng.normal(0, sigma, NPRE[i])) for i in range(4)}
        post = {f"A{i}": list(base[i] + ch[i] + rng.normal(0, sigma, NPOST[i])) for i in range(4)}
        _, d, _ = ak.paired_animal_draws(post, pre, mean_of, rng, 1000)
        lo, hi = np.percentile(d, [2.5, 97.5]); fb += (lo > 0) or (hi < 0)
        c = [np.mean(post[a]) - np.mean(pre[a]) for a in pre]; ft += stats.ttest_1samp(c, 0).pvalue < 0.05
    return fb / n_sim, ft / n_sim
for tau, sigma in [(0.0, 0.05), (0.0, 0.02), (0.05, 0.05), (0.05, 0.02), (0.02, 0.05)]:
    for eff in (0.0, 0.05, 0.10):
        b, t = sim(eff, tau, sigma, 400, 1)
        print(f"tau={tau} sigma={sigma} effect={eff}:  bootstrap star {b:.2f}   t-test(df3) star {t:.2f}", flush=True)

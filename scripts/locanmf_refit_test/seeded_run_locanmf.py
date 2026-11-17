"""`python -m wfield_local.run_locanmf` with numpy + torch seeded first (joint_locanmf._seed_everything).

Production per-session fits set NO seed; this wrapper exists only for the same-machine stability test.
    python seeded_run_locanmf.py <seed> <run_locanmf args...>
"""
import sys

from wfield_local import joint_locanmf, run_locanmf

if __name__ == "__main__":
    seed = int(sys.argv[1])
    joint_locanmf._seed_everything(seed)
    sys.argv = ["run_locanmf"] + sys.argv[2:]
    raise SystemExit(run_locanmf.main())

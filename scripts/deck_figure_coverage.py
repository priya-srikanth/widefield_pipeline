"""WHICH RENDERED FIGURES DOES THE DECK NEVER PLACE?

    PYTHONPATH=$(pwd) python scripts/deck_figure_coverage.py [figures_dir]

THE FAILURE THIS FINDS. A figure is produced by the nightly and referenced by no slide. It cannot
be detected downstream: the deck's figures-placed/missing counter only counts slides it TRIED to
build, and `if not path.exists(): continue` is silent by construction, so a filename the deck never
constructs at all is invisible from both ends. Two instances were found by hand on 2026-08-24 --
twelve behaviour slides (`coding_engagement_*`, the filename carried no method but the deck built one
into every path) and 144 per-session per-class matrices (`coding_crosssess_*`, `coding_pairsess_*`,
which no slide referenced at all) -- and finding them by hand is not a method.

HOW IT WORKS. Every string literal in the deck builder that looks like a PNG filename is turned into
a regex: literal fragments kept, `{...}` substitutions replaced by a wildcard. Files matching no
pattern are unreferenced. Also reported in reverse: patterns matching NO file, which are either dead
references or a renderer that stopped producing something.

WHAT IT CANNOT SEE. A pattern is matched textually, not evaluated -- so a glob the deck builds but
whose loop never reaches a given value (an animal missing from the config, a window not in the
tuple) still counts as "referenced". This finds families that are wholly absent, which is the failure
that has actually happened twice, not per-file placement.

READ THE "matching NO file" HALF WITH THE ROOT IN MIND. Only ONE figure root is scanned per run, so
every pattern belonging to a deck that reads a different root (the behaviour deck's `behavior_out`
tree, the preprocessing deck's per-session directories) shows up there and is not a dead reference.
Point the script at those roots separately to check them.
"""
import ast
import re
import sys
from pathlib import Path

#: "THE DECK" IS THREE DECKS. The analysis deck alone reports 700+ unreferenced files, almost
#: all of which the PREPROCESSING deck places -- per-date decoder/encoder/component panels are
#: its whole job. Checking one builder in isolation produces a scary number and no signal.
DECKS = [Path("wfield_local/locanmf_analysis_deck.py"),
         Path("wfield_local/preprocess_deck.py"),
         Path("wfield_local/behavior_deck.py")]
FIGDIR = Path(sys.argv[1] if len(sys.argv) > 1 else "E:/cue_lick")


def png_patterns(path):
    """Regexes for every PNG-looking string literal in the module, with {…} as a wildcard."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pats = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and ".png" in n.value:
            # DOCSTRINGS MENTION FILENAMES TOO, and a module docstring listing the layout it
            # assembles is not a reference. Anything multi-line or long is prose, not a path.
            if "\n" in n.value or len(n.value) > 120:
                continue
            pats.add(n.value)
        elif isinstance(n, ast.JoinedStr):
            parts, has_png = [], False
            for v in n.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(re.escape(v.value))
                    has_png = has_png or ".png" in v.value
                else:
                    parts.append("WILDCARD")
            if has_png:
                pats.add("".join(parts).replace("WILDCARD", "\x00"))
    out = []
    for p in pats:
        rx = p if "\x00" in p or "\\" in p else re.escape(p)
        rx = rx.replace("\x00", ".*").replace(r"\*", ".*").replace(r"\?", ".")
        out.append((p.replace("\x00", "{}"), re.compile("^" + rx + "$")))
    return out


pats = [q for d in DECKS if d.exists() for q in png_patterns(d)]
files = sorted(p.name for p in FIGDIR.glob("*.png"))
print(f"{len(files)} PNGs in {FIGDIR}, {len(pats)} filename patterns across "
      f"{len([d for d in DECKS if d.exists()])} deck builders\n")

unmatched, used = [], set()
for f in files:
    hit = [i for i, (_src, rx) in enumerate(pats) if rx.match(f)]
    if hit:
        used.update(hit)
    else:
        unmatched.append(f)


def family(name):
    """Collapse a filename to its family so 300 files report as a handful of lines."""
    s = re.sub(r"PS9[2345]", "<animal>", name)
    s = re.sub(r"\d{4}", "<date>", s)
    return s


# AGE IS THE DISCRIMINATOR, and without it this report is unusable. 137 unreferenced families
# sounds alarming and is mostly archaeology: figures from a June naming scheme, component
# figures superseded by a merged one, one-off grant panels. A file the CURRENT nightly still
# rewrites and no slide places is the actual gap -- that is how the 144 per-session matrices
# were distinguished from 700 files of history (2026-08-24).
newest = max((FIGDIR / f).stat().st_mtime for f in files) if files else 0
FRESH_DAYS = 3

fams = {}
for f in unmatched:
    fams.setdefault(family(f), []).append(f)


def age_days(fs):
    newest_in = max((FIGDIR / f).stat().st_mtime for f in fs)
    return (newest - newest_in) / 86400.0


fresh = {k: v for k, v in fams.items() if age_days(v) <= FRESH_DAYS}
stale = {k: v for k, v in fams.items() if age_days(v) > FRESH_DAYS}

print(f"--- {sum(len(v) for v in fresh.values())} files in {len(fresh)} CURRENT families the deck "
      f"never places (rewritten within {FRESH_DAYS} days of the newest figure)\n")
for fam, fs in sorted(fresh.items(), key=lambda kv: -len(kv[1])):
    print(f"  {len(fs):>4}  {fam}")
print(f"\n--- {sum(len(v) for v in stale.values())} files in {len(stale)} STALE families "
      f"(not rewritten by the current nightly -- archaeology, not a gap)\n")
for fam, fs in sorted(stale.items(), key=lambda kv: -age_days(kv[1]))[:12]:
    print(f"  {len(fs):>4}  {fam:<62} {age_days(fs):>6.0f} d older")
if len(stale) > 12:
    print(f"  ... and {len(stale) - 12} more")

dead = [src for i, (src, _rx) in enumerate(pats) if i not in used]
print(f"\n--- {len(dead)} patterns matching NO file (dead reference, or renderer stopped)\n")
for src in sorted(dead):
    print(f"  {src}")

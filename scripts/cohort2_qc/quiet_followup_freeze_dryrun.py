"""DRY RUN (load-only, writes nothing to any store): which frozen precue decoders quiet_task_events_followup.py needs.

11-25 (four-camera queue, IMAGING_FAILED_STEPS_20261125): the follow-up refused on PS92 ("no stored frozen precue
decoder for this spec -- refusing to fit (rule 10)"). This builds EXACTLY the follow-up's spec per animal
(`quiet_task_events_followup.frozen_models_for` up to `decoder_spec`, same env: the quiet-chain profile, DAQ lick
source), then reports per animal: the spec id / slug, whether the store has it (`frozen_models.find`), and every
stored sibling (same animal, kind, lick source, window) with the spec keys it differs in. Nothing is fitted, frozen,
renamed or written into the store; the report goes to stdout and to --report (a new file outside every store).

    PYTHONPATH=<wt>;<wt>/scripts python scripts/cohort2_qc/quiet_followup_freeze_dryrun.py --report <md>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ANIMALS = ("PS92", "PS93", "PS94", "PS95")
#: spec keys that are long lists / digests: reported as a count / prefix, not verbatim
LONG = {"train_labels", "train_sigs"}


def spec_for(animal):
    from wfield_local import config
    from wfield_local.locanmf_frozen_decoder import decoder_spec, pool_sessions
    from wfield_local.precue_engagement_states import features_with_indices

    import quiet_task_events_followup as F
    pre = [lab for lab in config.phase_labels("pre") if lab.startswith(animal)]
    post = [lab for lab in config.phase_labels("post") if lab.startswith(animal)]
    ft = features_with_indices(F._basis(animal), nolick_ref="cue")
    XE, YE, GE, _BE, _XU, _YU, kept, _c, _GU = pool_sessions(pre + post, source="locanmf", align="precue", post_s=2.0,
                                                             features=ft)
    pre_list = [i for i, lab in enumerate(kept) if lab in set(pre)]
    return decoder_spec(XE, kept, pre_list, YE=YE, GE=GE, align="precue", source="locanmf", post_s=2.0,
                        basis_id=getattr(ft, "basis_id", None)), [kept[i] for i in pre_list]


def _short(k, v):
    if k in LONG and isinstance(v, (list, tuple)):
        return f"[{len(v)}] " + ",".join(map(str, v[:12])) + (" ..." if len(v) > 12 else "")
    s = str(v)
    return s if len(s) <= 80 else s[:77] + "..."


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--animals", nargs="*", default=list(ANIMALS))
    a = ap.parse_args(argv)
    os.environ["WIDEFIELD_FROZEN_LOAD_ONLY"] = "1"
    from wfield_local import frozen_models as fm
    lines = ["# quiet_task_events_followup: frozen precue decoders needed (DRY RUN, nothing frozen)", "",
             f"store (local_dir): `{fm.local_dir()}`; lick source `{os.environ.get('WIDEFIELD_LICK_SOURCE')}`; "
             f"rebuild root `{os.environ.get('WIDEFIELD_REBUILD_ROOT')}`", ""]
    need = []
    for an in a.animals:
        spec, pre = spec_for(an)
        d, origin = fm.find(spec)
        slug = fm._slug(spec)
        lines += [f"## {an}", "", f"- spec id `{fm.spec_id(spec)}`, dir name `{slug}`",
                  f"- stored: **{'YES ' + str(d) + ' (' + str(origin) + ')' if d else 'NO -> would be a NEW freeze'}**",
                  f"- training (pre) sessions {len(pre)}: {', '.join(pre)}",
                  "- spec: " + "; ".join(f"{k}={_short(k, v)}" for k, v in sorted(spec.items()) if k not in LONG), ""]
        if not d:
            need.append(an)
        sib = fm.siblings(spec)
        lines.append(f"- stored siblings (same animal / kind / lick source / window): {len(sib)}")
        for s in sib:
            diff = sorted(k for k in set(s["spec"]) | set(spec) if str(s["spec"].get(k)) != str(spec.get(k)))
            lines.append(f"  - `{s['dir'].name}` frozen {s['frozen_utc']} ({s['origin']}): differs in "
                         + ", ".join(f"{k} (stored {_short(k, s['spec'].get(k))} / needed {_short(k, spec.get(k))})"
                                     for k in diff))
        lines.append("")
        print("\n".join(lines[-8:]), flush=True)
    lines += ["## To freeze (needs Priya)", "",
              f"{len(need)} new entries: " + (", ".join(need) or "none"),
              "Command once approved (from the same env, after this list is accepted): the follow-up itself does NOT "
              "fit (its fitter refuses); the entries come from attempt_coding's precue freeze under this key, "
              "`WIDEFIELD_LICK_SOURCE=daq python -m wfield_local.attempt_coding --aligns precue --freeze` with the "
              "freeze switch of the code that runs it (main: --allow-freeze / WFIELD_ALLOW_FREEZE=1) (check that its printed spec ids equal the ones above before accepting).", ""]
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"need": need}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

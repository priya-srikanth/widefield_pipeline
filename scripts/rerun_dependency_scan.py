"""List every module / script that reads an input the cam2 / cam3 re-run changes (3-D tables, camera lick trains,
camera trial classes, the cam4 spout frame), directly and through wfield_local imports. Prints CSV to stdout; writes
nothing. Used by docs/status/STATUS_2026-11-18_CAM23_RERUN_INVENTORY.md (refresh its appendix with
``python scripts/rerun_dependency_scan.py > docs/status/cam23_rerun_20261118/dependency_scan.csv``).

Tags: 3D = pose3d tables (session_poses/pose3d, pose3d_4cam); LICK = LP / cam1 lick trains, licks_all / licks.csv,
RESTnl mask; CLASS = trial_classes / trial_attempts / working gate; SPOUT = the cam4 2-D spout frame and kinematics
chain. A regex scan, so it over-includes (QC and review scripts) rather than misses."""
import re,pathlib,collections
files=[p for p in pathlib.Path('.').rglob('*.py') if not any(x in p.parts for x in ('archive','tests','.git','notebooks'))]
imp={}
for p in files:
    s=p.read_text(encoding='utf-8',errors='ignore')
    names=set()
    for sub,lst in re.findall(r'from wfield_local(?:\.(\w+))? import ([\w, ()\n]+)',s):
        if sub: names.add(sub)
        else: names|=set(re.findall(r'\w+',lst))
    names|=set(re.findall(r'wfield_local\.(\w+)',s))
    if re.search(r'pose3d_dir|session_poses|pose3d_4cam',s): names.add('@3d_tables')
    if re.search(r'licks_all\.csv|licks\.csv',s): names.add('@cam4_lick_csv')
    if re.search(r'cam1_licks|_events\.csv|events_all',s): names.add('@cam1_licks')
    imp[p.as_posix()]=names
ROOTS={'pose3d':'3D','pose3d_side':'3D','@3d_tables':'3D','lp_licks':'LICK','@cam1_licks':'LICK','@cam4_lick_csv':'LICK','restnl_mask':'LICK','trial_classes':'CLASS','trial_attempts':'CLASS','working_trials':'CLASS','gate_source':'CLASS','spout_frame':'SPOUT','orofacial_outputs':'SPOUT','tongue_kinematics':'SPOUT','kinematics_chain':'SPOUT'}
def mod_of(f): return f[len('wfield_local/'):-3] if f.startswith('wfield_local/') else None
tag=collections.defaultdict(set)
for f,n in imp.items():
    for r,t in ROOTS.items():
        if r in n: tag[f].add(t)
direct={f:set(t) for f,t in tag.items()}
changed=True
while changed:
    changed=False
    for f,n in imp.items():
        for g in imp:
            m=mod_of(g)
            if m and m in n and not tag[g]<=tag[f]:
                tag[f]|=tag[g]; changed=True
print("file,direct,transitive")
for f in sorted(imp):
    if tag[f]: print(f"{f},{'+'.join(sorted(direct.get(f, set())))},{'+'.join(sorted(tag[f]))}")

# Status and handoff docs — what each one settled

Seventeen dated snapshots, 7,303 lines, written at the end of a working day or before a context
compaction. **They are the project's memory, and none of them has been deleted.** Several record
a number that was later withdrawn, or a plan that turned out wrong; those are kept unedited,
because knowing what was believed that week is how a stale figure in a document gets traced back
to the run that produced it.

Consolidated into this directory on 2026-09-21 — `docs/` was sixteen status files deep and the
topic docs were hard to see between them.

**Reading rule: newest first, and each one says what it supersedes.** The chain is real — most of
these open by naming the file they replace and what that file remains the record of. Start at the
bottom of this table and walk up only as far as you need.

| date | lines | what it settled | superseded by |
|---|---:|---|---|
| [08-14](STATUS_2026-08-14.md) | 219 | the drift-removal decision week. **Carries two withdrawn claims**, marked in the file: the +0.032 pre-cue sub-binning gain did not replicate (+0.009, 23/44) | 08-17 |
| [08-09 tasks](STATUS_2026-08-09_TASKS.md) | 88 | **the open-items list, retired 2026-09-22** — it had stopped being open. Kept as the record of what was outstanding in August; its header says where each live fact went | — |
| [08-17](STATUS_2026-08-17.md) | 299 | **the seventeen silent bugs found 15–17 Aug** and the guards put in for each. Those guards are still in force | 08-23 |
| [08-23](STATUS_2026-08-23.md) | 140 | the post-stroke readout's design and the nulls behind it | 08-26 |
| [08-26](STATUS_2026-08-26.md) | 198 | cohort state after the lesion series | 08-28 |
| [08-28 handoff](STATUS_2026-08-28_handoff.md) | 218 | five approved items, all since done — kept as the record of **what the plan got wrong** | — |
| [08-28 evening](STATUS_2026-08-28_evening.md) | 94 | what was running that night | — |
| [08-28 night](STATUS_2026-08-28_night.md) | 153 | the pooled epoch figures, the engagement gate, and the parallel render | — |
| [09-07](STATUS_2026-09-07.md) | 226 | both engagement gates **backdated**, and the cue-aligned clip pipeline built on the trial classes they define | — |
| [09-10](STATUS_2026-09-10.md) | 139 | state before a compaction; the science is in [`PRELIM_DATA_VLS_STROKE.md`](../PRELIM_DATA_VLS_STROKE.md) | — |
| [09-12](STATUS_2026-09-12.md) | 605 | the anatomical arm | — |
| [09-13](STATUS_2026-09-13.md) | 2,618 | **the REST baseline migration** — the largest of these, on branch `worktree-rest-baseline` | partly 09-16 |
| [09-16](STATUS_2026-09-16.md) | 522 | the engagement-gate audit and the `restdock05` migration. **Cited most often**; `CLAUDE.md` points here | 09-17, for everything after the rest-arm audit |
| [09-17](STATUS_2026-09-17.md) | 1,145 | the rest-arm audit and what followed it | — |
| [09-19](STATUS_2026-09-19.md) | 254 | **the 415 nm day** and the engagement decomposition | 09-20 |
| [09-20](STATUS_2026-09-20.md) | 127 | haemodynamics closed, engagement standing, loops parallelised | — |
| [09-21 engineering](STATUS_2026-09-21_ENGINEERING.md) | 258 | **the codebase-quality plan** — module splits, output-tree restructure, the archive pass. Sections 2.1–2.5 | — |

---

## Which document answers which question

These four kinds of document are easy to confuse, and looking in the wrong one is how a
settled question gets re-litigated:

| you want | read |
|---|---|
| **why** something is the way it is | [`DECISIONS.md`](../../DECISIONS.md) — the reasoning, usually because the alternative was tried and failed |
| **what the code is** and how the parts relate | [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) |
| **what to run**, tonight or for the first time | [`runbooks/`](../../runbooks) |
| **what was true on a given day** | here |

A status doc is a snapshot and goes out of date by design. `DECISIONS.md` is cumulative and does
not. **If the two disagree, `DECISIONS.md` wins** — and that disagreement is worth a line in
`DECISIONS.md` saying so.

# Helper-box setup (a THIRD machine, no local raw)

For running one animal's preprocessing / LocaNMF on a spare workstation when the imaging box is
busy. First used 2026-08-11 on the **Priya lab desktop** for PS95 8/10. Findings behind these steps:
`DECISIONS.md` ("Running preprocessing on a THIRD (helper) box", Part II LocaNMF).

The box mounts MICROSCOPE at `N:` and standby at `M:` and has **no local raw drive**.

---

## 1. Path resolution (required — nothing works without it)

`paths.detect_machine()` looks for a signature mount (`E:\labcams_data` -> imaging,
`M:\MICROSCOPE\Priya` -> analysis). A helper box matches neither, so it silently defaults to
`analysis` and resolves every root to a non-existent `M:\MICROSCOPE\...`. Fix without editing
`configs/`:

```powershell
subst E: C:\wf_local            # E:\labcams_data\<DATE>\<session>\raw_widefield_data\ , E:\DAQ_recorder_output\
$env:WIDEFIELD_MACHINE = "imaging"
```

**Neither survives a reboot.** Re-apply both before any run. Verify:

```powershell
python -c "from wfield_local.paths import PathResolver as P; r=P(); print(r.machine, r.root('labcams'))"
# -> imaging N:/MICROSCOPE/Priya/Widefield/labcams
```

Then stage that session's raw `.dat` + DAQ `.h5` under `E:` and run
`python -m wfield_local.preprocess <DATE> --only <ANIMAL>` as normal. The ~190 GB `.bin` stays on
local scratch; only results are pushed to `N:`.

## 2. Two footguns when only ONE animal is being processed here

* **Never run `preprocess_deck` / `build_decks`.** It globs `cross-session_preprocessing*.pptx` and
  **deletes every sibling deck it did not write this run**, destroying the other animals' decks that
  the imaging box is building. Call `build_deck` (singular) with `sessions` filtered to the animal.
* **Skip the photobleach step** (`--skip-photobleach`). `photobleach.run()` calls `summary()`, which
  rewrites the date's **shared** `photobleach_SUMMARY.png` + `photobleach_results.json` with only the
  animals in *this* run. Call `photobleach.analyze()` alone — the deck only reads the per-session
  `photobleach_<ANIMAL>_<MMDD>.png`.

`refresh_xall` (per-animal) and `crossday_intensity` (whole-tree rollup, improved by re-running after
the push) are both safe.

## 3. Transfers off MICROSCOPE are unreliable

The `N:` mount drops mid-transfer (`ERROR 53 -- the network path was not found`). Plain `robocopy`
**discards a partial 190 GB file** on a drop. Use a resumable copier that hashes in the same pass,
and always verify the raw `.sha256` sidecar. Copy to/from the **UNC path**, not the drive letter —
the mapped letter is torn down by session events that leave the UNC reachable.

Never delete local staging until the server copy is byte-verified (`DECISIONS.md` rule 1).

## 4. GPU (optional)

LocaNMF runs without the compiled extension (pure-PyTorch fallback). Building `cuhals` gave **~6x**
(9.3 min vs ~55 min on an RTX 5060). Full walkthrough: [`../docs/GPU_SETUP.md`](../docs/GPU_SETUP.md).

---

## 5. Windows Update suspended on this box (2026-08-11)

Long jobs here run for hours (a 190 GB verified transfer, ~1 h; a full cross-session recompute, ~1.5 h),
and an unattended Windows Update reboot kills them **and** can leave a half-written file on MICROSCOPE.
Automatic updates were therefore disabled on the Priya lab desktop:

```powershell
# RUN AS ADMINISTRATOR (HKLM writes need elevation)
$au = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU'
if (-not (Test-Path $au)) { New-Item $au -Force | Out-Null }   # NB: -Force on an EXISTING key wipes its values
Set-ItemProperty $au NoAutoUpdate 1 -Type DWord                    # disable automatic updates
Set-ItemProperty $au NoAutoRebootWithLoggedOnUsers 1 -Type DWord   # never force-reboot while logged in
```

This box is **workgroup-joined with no third-party MDM**, so nothing pushes the setting back — it
persists until manually reverted. (On a domain- or Intune-managed machine it would be overwritten at
the next policy refresh.)

**This box therefore stops receiving security patches.** It holds write credentials to two
institutional file servers, so treat it as a temporary measure and re-enable when the run window
closes.

### Undo (restore normal Windows Update)

```powershell
# RUN AS ADMINISTRATOR
Remove-Item 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate' -Recurse -Force
gpupdate /force
Start-Service wuauserv
```

Then check for updates once manually (Settings -> Windows Update) to catch up.

### Check current state (no elevation needed)

```powershell
Get-ItemProperty 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU' -EA SilentlyContinue |
  Select-Object NoAutoUpdate, NoAutoRebootWithLoggedOnUsers
(Get-Service wuauserv).Status
```

`NoAutoUpdate = 1` and `wuauserv = Stopped` means updates are currently suspended.

### Less drastic alternatives

If the goal is only "don't reboot during a long run", these keep patching alive:

```powershell
$ux = 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings'
Set-ItemProperty $ux PauseUpdatesExpiryTime (Get-Date).AddDays(35).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
Set-ItemProperty $ux ActiveHoursStart 7  -Type DWord
Set-ItemProperty $ux ActiveHoursEnd   23 -Type DWord
```

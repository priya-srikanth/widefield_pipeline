# GPU setup for LocaNMF (optional `cuhals` fast path)

LocaNMF runs on **any** machine with a CUDA GPU and PyTorch. The compiled `cuhals` extension is a
**speed optimization, not a requirement** — `locanmf/demix.py` wraps `import cuhals` in
`try/except ImportError` and falls back to `native_update`, a pure-PyTorch HALS that still runs on
the GPU. Build it only if the box will run LocaNMF regularly.

Measured on the Priya lab desktop (RTX 5060, PS95 8/9, r2 0.95 / loc 80 / maxrank 20):

| path | wall clock | components |
|---|---|---|
| pure-torch fallback (`use_cuhals = False`) | ~55 min | 164 |
| compiled `cuhals` (`use_cuhals = True`)    | **9.3 min** | 167 |
| reference (analysis box, cuhals)           | —        | 160 |

**~6x faster**, and the component counts agree to within the rank line search's own jitter (it is a
greedy search, so exact equality is not expected; 54/64 Allen regions matched exactly in a fuller
comparison — see `DECISIONS.md` Part II).

---

## 0. Check what you already have

```powershell
nvidia-smi                                    # GPU + driver; note the architecture
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_arch_list())"
```

**Match the CUDA minor version to the torch build.** Blackwell (RTX 50-series) is `sm_120` and needs
**cu128** — the `cu124` build named in the older kickoff doc does **not** support it:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

If `torch.cuda.get_arch_list()` does not contain your GPU's `sm_XX`, stop and fix that first; nothing
below will help.

---

## 1-3. Prerequisites (in this order — all need admin)

1. **Visual Studio 2022 Build Tools** + the C++ workload (provides `cl.exe`):
   ```powershell
   winget install --id Microsoft.VisualStudio.2022.BuildTools `
     --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   ```
2. **CUDA Toolkit matching torch** (torch cu128 -> CUDA 12.8). Install *after* VS so it registers its
   MSBuild integration:
   ```powershell
   winget install --id Nvidia.CUDA --version 12.8
   ```
3. **MKL into the conda env.** `setup.py` (as patched) links `mkl_rt` and includes
   `$CONDA_PREFIX\Library\include`. A pip-only env has **no MKL** (numpy ships `scipy-openblas`), so
   the LINK step fails with a missing `mkl_rt.lib`:
   ```powershell
   conda install -n locanmf -c conda-forge mkl mkl-devel
   ```

---

## 4. Clone + patch + build

```powershell
git clone https://github.com/ikinsella/locaNMF C:\Users\<you>\Github\locaNMF
```

Apply **both** patches from `wfield_local/`:

* `locanmf_torch_compat.patch` — 3 edits for modern torch (`out=` may not alias its input; bool masks).
  **`git apply` may reject it** (EOF/whitespace drift in `factor.py`); the edits are small, apply by hand.
* `locanmf_cuhals_win_build.patch` — MSVC/Windows build (setup.py include/lib dirs, `/openmp`, `mkl_rt`,
  `-allow-unsupported-compiler`). This one applies cleanly.

Then run the build script (kept next to the clone; it pre-checks vcvars/nvcc/MKL and fails with a clear
message instead of a compiler error):

```powershell
cmd /c C:\Users\<you>\Github\locaNMF\build_cuhals.bat
```

### Three traps this script exists to avoid

1. **`DISTUTILS_USE_SDK=1` is mandatory.** With the VC environment active, torch's `cpp_extension`
   aborts: *"the VC environment is activated but DISTUTILS_USE_SDK is not set"*.
2. **Extend `PATH` BEFORE calling `vcvars64.bat`.** In a compound `cmd /c "a && b"` line, `%PATH%`
   expands once at parse time, so setting `PATH` *after* vcvars silently discards the compiler paths it
   just added — and the build dies with `error: command 'cl.exe' failed: None` even though `cl.exe` is
   installed.
3. **`pip install .` does not forward `--with-extension`.** `setup.py` detects it via `sys.argv`, which
   the modern build backend does not pass through, so pip silently builds **without** the extension.
   Use `python setup.py install --with-extension`.

## 5. Verify

```powershell
python -c "import torch, cuhals; from locanmf.demix import use_cuhals; print('use_cuhals =', use_cuhals)"
```

`use_cuhals = True` means the fast path is live.

**`import cuhals` on its own will fail** with `ImportError: DLL load failed` even on a perfect build —
the extension links against torch's DLLs (`c10`, `torch_cpu`), and those directories are only added to
the DLL search path by `import torch`. Always import torch first. `locanmf/demix.py` already does.

## Falling back

Nothing in the pipeline depends on `cuhals`. To drop back to the pure-torch path,
`pip uninstall locanmf` and reinstall without `--with-extension`; `use_cuhals` returns to `False` and
runs continue at the slower speed with equivalent results.

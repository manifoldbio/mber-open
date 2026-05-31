# Local Modifications — mber-open

All changes are local to `hongjunbai/mber-open`. Upstream: `manifoldbio/mber-open`.

| Commit | Description |
|--------|-------------|
| `344c7fe` | Prevent GPU OOM: force PLM/folding models to CPU, configurable overrides |

**Path abbreviations** (relative to project root):
- `core/` → `src/mber/core/`
- `models/` → `src/mber/models/`
- `utils/` → `src/mber/utils/`
- `VHH/` → `protocols/src/mber_protocols/stable/VHH_binder_design/`

---

## Memory Management — Reusable Module Lifecycle + GPU OOM Prevention

**Problem (Memory Leak)**: Upstream creates fresh TrajectoryModule and EvaluationModule for
each trajectory. This causes repeated JAX JIT compilation + model reload across trajectories,
which accumulates unreclaimable GPU memory (JIT cache + TracerState leak in JAX).

**Problem (OOM)**: ESM2 and NBB2 also try to load on GPU. Combined with the AF2 model, this causes OOM on most consumer grade GPUs (< 24GB VRAM).

**Changes — Module reuse (fixes memory leak)**:
- `VHH/cli.py` — `main()` creates `TrajectoryModule` and `EvaluationModule` ONCE, passes to
  `_run_one_trajectory()`. TemplateModule remains fresh per trajectory — each trajectory
  re-generates the initial binder sequence from a new ESM sample, producing a different
  starting prior for optimization.
- `core/modules/trajectory.py` — `self._precompile_done` flag: JIT precompile runs only on first trajectory
- `core/modules/evaluation.py:_setup_models()` — `af_complex_model`, `esm_model`, and
  `monomer_folding_model` guarded with `if self.<model> is None:` checks so model weights load
  only on the first setup call. Subsequent calls skip instantiation but still reconfigure
  loss and evaluation parameters for the new design state.

**Changes — GPU allocation**:
- `core/modules/template.py` — ESM2 and folding model (NBB2/ESMFold, monomer only) load with configurable device (`plm_force_cpu`, `folding_force_cpu`)
- `models/folding/nbb2_model.py` — NBB2 ignores the `device` arg and forces GPU via `torch.cuda.is_available()`.
  Monkey-patched during init to force CPU weight load, then moved to `self.device`. Also
  defaults `device="cpu"` instead of auto-detecting GPU.
- `core/modules/evaluation.py` — `folding_force_cpu` applied to monomer folding model (NBB2/ESMFold)

**Memory behavior (before vs after)**:

| Aspect | Before (upstream) | After (local) |
|--------|-------------------|---------------|
| TrajectoryModule | fresh per trajectory | created once, reused |
| EvaluationModule | fresh per trajectory | created once, reused |
| Model instantiation | every trajectory | first trajectory only |
| AF2 JIT tracing | every trajectory | first trajectory only |
| JAX TracerState | accumulates (leak) | bounded (no leak) |
| ESM2 model device | GPU | CPU (default) |
| NBB2 model device | GPU (ignores device="cpu") | CPU (default) |

**Result**: First trajectory pays the model loading and JIT compilation cost. Subsequent trajectories
reuse the warmed-up models and their cached traces, skipping weight loading while reconfiguring
loss and evaluation parameters. GPU memory stays bounded across the full run. Non-critical
ESM2 and folding models offloaded to CPU.

### How to Use

```bash
# Default: PLM + folding models on CPU (safe for GPUs with 24GB VRAM)
mber-vhh --input-pdb target.pdb --output-dir out --chains A --hotspots A56

# Override: put PLM/folding models on GPU (may requires >24GB VRAM)
mber-vhh --input-pdb target.pdb --output-dir out --no-plm-force-cpu --no-folding-force-cpu
```

In settings YAML:
```yaml
template:
  plm_force_cpu: true      # default
  folding_force_cpu: true   # default

evaluation:
  plm_force_cpu: true
  folding_force_cpu: true
```

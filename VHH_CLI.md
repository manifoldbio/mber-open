## mBER VHH CLI – quick start

This guide shows how to run the VHH binder design protocol using a simple CLI, either with a settings file, flags, or an interactive prompt.

### Prerequisites
- Install the protocols package (adds the `mber-vhh` command):
```bash
# from repo root
pip install -e protocols
```
- Download AlphaFold params (if not already):
```bash
bash ./download_af_weights.sh
```

### Option 1: Use the example settings file (recommended)
An example PDL1 target and settings file are included. You can run them as-is:
```bash
mber-vhh --settings ./protocols/src/mber_protocols/examples/vhh_settings_example.yml
```
The example references:
- Target PDB: `./protocols/src/mber_protocols/examples/PDL1.pdb`
- Output: `./output/vhh_pdl1_A56`

### Option 2: Flags-only (no file)
When not using `--settings`, three flags are required: `--input-pdb`, `--output-dir`, `--chains`. All others are optional and have sensible defaults.
```bash
mber-vhh \
  --input-pdb ./protocols/src/mber_protocols/examples/PDL1.pdb \
  --output-dir ./output/vhh_pdl1_A56 \
  --chains A \
  --hotspots A56 \
  --num-accepted 100 \
  --max-trajectories 10000 \
  --min-iptm 0.75 \
  --min-plddt 0.70
```
- Required (if not using `--settings`): `--input-pdb`, `--output-dir`, `--chains`
- Optional (defaults in parentheses): `--hotspots` (none), `--num-accepted` (100), `--max-trajectories` (10000), `--min-iptm` (0.75), `--min-plddt` (0.70)

### Option 3: Interactive mode
Prompts for the same inputs with defaults shown:
```bash
mber-vhh --interactive
```
Prompts include example formats (e.g., hotspots like `A56` or `A56,B20`) and default values:
- num_accepted: 100
- max_trajectories: 10000
- min_iptm: 0.75
- min_plddt: 0.70

### Settings file schema (minimal)
YAML/JSON with only a few keys; all VHH protocol defaults remain unchanged:
```yaml
output:
  dir: /abs/path/out
target:
  pdb: /abs/path/target.pdb          # or PDB code, UniProt ID, s3://...
  chains: "A"                        # e.g., "A" or "A,B"
  hotspots: ["A56"]                  # optional; omit or [] to bind anywhere
stopping:
  num_accepted: 100
  max_trajectories: 10000
filters:
  min_iptm: 0.75
  min_plddt: 0.70
```

### Outputs
Given `output_dir`, the CLI writes:
- `accepted.csv` – sequence and metrics for accepted designs
- `Accepted/` – complex/relaxed PDBs for accepted binders
- `runs/<trajectory_name>/` – full state for each trajectory (JSONs, PDBs, summary)

### Notes
- Hotspots are optional; omit to bind anywhere.
- Chains accepts single or multiple chains (e.g., `A` or `A,B`).
- All VHH protocol defaults are preserved (losses, iterations, optimizer, models, etc.).
- Device defaults to GPU if available (standard mBER defaults). Ensure your CUDA device is visible.

### Help
```bash
mber-vhh -h
```


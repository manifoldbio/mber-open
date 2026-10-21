# Third-party notices

mBER's own code is released under the MIT License (see [`LICENSE`](LICENSE)). Parts of
this repository come from other projects and stay under their original licenses. This
file lists them, reproduces the notices their licenses require, and records what mBER
downloads or depends on.

| Component | License | Where |
|---|---|---|
| [AlphaFold](#alphafold) | Apache-2.0 | `src/mber/models/alphafold/` |
| [ColabDesign](#colabdesign) | Beerware | parts of `src/mber/models/colabdesign/{design,prep,loss}.py` |
| [ColabFold](#colabfold) | MIT | `src/mber/models/colabfold/relax.py` |
| [BindCraft](#bindcraft) | MIT | parts of `src/mber/models/colabdesign/loss.py`; AlphaFold2 download steps in `download_weights.sh` |
| [Example and test structures](#example-and-test-structures) | CC0 1.0 (PDB data) | `protocols/.../examples/PDL1.pdb`; AlphaFold `testdata/` |
| [OpenStructure stereochemical parameters](#openstructure-stereochemical-parameters-downloaded-not-bundled) | LGPL-3.0 | downloaded, not bundled |
| [Python and conda dependencies](#dependencies-installed-separately-not-bundled) | various | installed separately, not bundled |
| [Model weights](#model-weights-downloaded-not-bundled) | CC BY 4.0, MIT | downloaded, not bundled |

---

## AlphaFold

- Source: https://github.com/google-deepmind/alphafold, the `alphafold/` package as of
  commit `020cd6d` (2025-03-17), including its tests, test data and notebook helpers.
- License: Apache License 2.0, Copyright 2021 DeepMind Technologies Limited. The full text
  is in [`src/mber/models/alphafold/LICENSE`](src/mber/models/alphafold/LICENSE), and every
  file keeps its original header.
- Changes made by Manifold Bio (each changed file says so under its license header):
  - internal imports point to `mber.models.alphafold` instead of `alphafold`, and some files
    were reformatted;
  - `relax/amber_minimize.py` passes the minimisation tolerance in the units OpenMM >= 8.4
    expects;
  - `common/residue_constants.py` fetches `stereo_chemical_props.txt` instead of bundling it
    (see below).

## ColabDesign

- Source: https://github.com/sokrypton/ColabDesign at commit `d024c4e` (also an installed
  dependency).
- In mBER: `_mutate` in `src/mber/models/colabdesign/design.py`, `_prep_binder` in
  `src/mber/models/colabdesign/prep.py`, and the helix-loss computation in
  `src/mber/models/colabdesign/loss.py` are adapted from ColabDesign.
- License (ColabDesign `LICENSE.txt`, reproduced as it requires):

```
/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * <so3@mit.edu> wrote this file.  As long as you retain this notice you
 * can do whatever you want with this stuff. If we meet some day, and you think
 * this stuff is worth it, you can buy me a beer in return.  Sergey Ovchinnikov
 * ----------------------------------------------------------------------------
 */
```

ColabDesign's bundled AlphaFold code (`colabdesign/af/alphafold`) is Apache-2.0
(DeepMind); its bundled ProteinMPNN weights are MIT (Justas Dauparas).

## ColabFold

- Source: https://github.com/sokrypton/ColabFold, `colabfold/relax.py`.
- In mBER: `src/mber/models/colabfold/relax.py`. Manifold Bio changed the import paths, and
  `relax_me` returns `(pdb, debug_data, violations)` instead of only the PDB.
- License: MIT.

```
MIT License

Copyright (c) 2021 Sergey Ovchinnikov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## BindCraft

- Source: https://github.com/martinpacesa/BindCraft (`functions/colabdesign_utils.py`,
  `install_bindcraft.sh`).
- In mBER: `add_rg_loss`, `add_i_ptm_loss`, `add_helix_loss` and
  `add_termini_distance_loss` in `src/mber/models/colabdesign/loss.py`; the AlphaFold2
  parameter download steps in `download_weights.sh`.
- License: MIT.

```
MIT License

Copyright (c) 2024 Martin Pacesa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Example and test structures

- `protocols/src/mber_protocols/stable/VHH_binder_design/examples/PDL1.pdb`: chain A of
  PDB entry 5JDS (https://www.rcsb.org/structure/5JDS).
- AlphaFold's `testdata/` structures (2RBG, 1GCN and others) come with AlphaFold (above).
- PDB data is available under CC0 1.0 (https://www.rcsb.org/pages/usage-policy).

## OpenStructure stereochemical parameters (downloaded, not bundled)

`stereo_chemical_props.txt` (Engh and Huber bond lengths and angles) is part of
OpenStructure, which is licensed under the GNU Lesser General Public License v3.0. mBER does
not include it. Like AlphaFold's Dockerfile and OpenFold's install script, mBER fetches it
instead: from ColabFold's byte-identical copy, which ships with OpenStructure's LGPL text
(https://github.com/sokrypton/ColabFold, commit `136bec6`, `colabfold/openstructure/`),
falling back to OpenStructure commit `7102c63615b64735c4941278d92b554ec94415f8`, and verifies
its SHA-256:
`download_weights.sh` saves it to the weights directory, the Docker build places it next to
`residue_constants.py`, and otherwise it is downloaded on first use. Set
`MBER_STEREO_CHEMICAL_PROPS` to use a local copy.

## Dependencies (installed separately, not bundled)

mBER does not include its dependencies; `environment.yml` and `requirements.txt` install
them from conda-forge, bioconda, PyPI and GitHub. Their licenses are listed here so they can
be tracked. To check the licenses actually installed in an environment, run
`python tools/dependency_licenses.py` (direct dependencies) or
`python tools/dependency_licenses.py --all` (everything installed).

| Dependency | License |
|---|---|
| torch, torchvision | BSD-3-Clause |
| torchaudio | BSD-2-Clause |
| transformers | Apache-2.0 |
| colabdesign (git, `d024c4e`) | Beerware (bundles Apache-2.0 AlphaFold code) |
| jax, flax | Apache-2.0 |
| ablang2 | BSD-3-Clause |
| ImmuneBuilder | BSD-3-Clause |
| freesasa | MIT |
| mdtraj | LGPL-2.1-or-later |
| biopython | Biopython License Agreement or BSD-3-Clause |
| biotite, numpy, ipywidgets, ipykernel, nbconvert, ipython | BSD-3-Clause |
| logomaker, pandas, pyyaml (mber-protocols) | MIT |
| matplotlib | Matplotlib License (PSF-style) |
| pyarrow, boto3 | Apache-2.0 |
| openmm (conda) | MIT; LGPL-3.0 for its CUDA and OpenCL platforms |
| pdbfixer (conda) | MIT |
| anarci (conda) | BSD-3-Clause (uses HMMER, BSD-3-Clause; its germline data is derived from IMGT, CC BY 4.0) |
| ffmpeg (conda) | LGPL-2.1+ or GPL-2.0+ depending on the build; used as a separate program for animations |

The GPU builds of torch and jax download NVIDIA CUDA libraries (cuBLAS, cuDNN and others),
which are distributed under NVIDIA's own license agreements rather than open-source licenses.

## Model weights (downloaded, not bundled)

`download_weights.sh` downloads model weights from their publishers; they are not part of
this repository.

| Weights | Source | License |
|---|---|---|
| AlphaFold2 parameters (2022-12-06) | https://storage.googleapis.com/alphafold/ | CC BY 4.0 (DeepMind) |
| NanoBodyBuilder2 weights | Zenodo record 7258553 | CC BY 4.0 (Abanades et al.) |
| ESM-2 650M, ESMFold (optional) | Hugging Face `facebook/esm2_t33_650M_UR50D`, `facebook/esmfold_v1` | MIT (Meta) |

If you redistribute these weights, for example in a Docker image built with
`INCLUDE_WEIGHTS=true`, follow their license terms (CC BY 4.0 requires attribution).

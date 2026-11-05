# EpNet-spd

**A learned empirical pseudopotential for electronic band structures across the *s*, *p* and *d* blocks.**

EpNet predicts the band structure of a crystal from its structure alone. It assembles a plane-wave Hamiltonian from
atom-centred operators: a local potential for every atom (an element term, an environment term from an equivariant
network, and a Coulomb term from learned charges), Kleinman–Bylander projectors for *s*/*p* atoms, and ultrasoft
*s*, *p*, *d* channels built from the all-electron atom for *d* atoms. One diagonalization per **k** point gives the
bands, with no self-consistency and no DFT code. The model has 78,964 parameters, covers 63 elements, and was trained
end to end through the eigensolver on Materials Project PBE band energies alone.

- **Web application:** <https://mmi.charlotte.edu/epnet> (upload a structure, get its bands and DOS).
- **Paper:** Q. Zhu, *A learned pseudopotential for electronic band structures across the s, p and d blocks*
  (manuscript in preparation). The environment encoder is that of EpNet-Si: Q. Zhu, arXiv:2610.14992 (2026).

| | Result (final model) |
|---|---|
| Held-out chemical systems (structures of up to 24 sites; mean of the two seeds) | median band error 0.061 eV within ±2 eV of the gap or Fermi level; median gap error 0.102 eV |
| 600,617 Alexandria PBE structures absent from the Materials Project (seed 0) | median gap error 0.108 eV; 98.9% correctly metal or insulator |
| Scope | non-magnetic, no spin–orbit coupling, no +U; valence bands (semicore states are not modelled) |

## Install

Python 3.9 or newer.

    git clone <this repository> && cd EpNet-prototype
    pip install -e .              # the model; add [app] for the web app, [data] for the Materials Project download, [test]

For a CPU-only machine, install the CPU build of PyTorch first
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`).

## Use

**Command line.** A CIF or POSCAR (or extxyz, Quantum ESPRESSO input, FHI-aims geometry.in, ...):

    python scripts/predict.py apps/epnet_bands/examples/TiO2_mp-2657.cif --out out/
    python scripts/predict.py my_structure.cif --workers 4      # k points over 4 processes (many-core CPUs)

The structure is reduced to its standardized primitive cell; bands follow the Setyawan–Curtarolo path, with the energy
zero at the valence-band maximum (insulators) or the Fermi level (metals); the density of states uses a Γ-centred mesh
reduced by symmetry. Outputs: bands CSV, DOS CSV, PNG, a JSON summary (gap, direct or indirect, band edges) and the
standardized cell.

**Python** (from the repository root):

```python
from apps.epnet_bands.core import predict, summary
p = predict(open("structure.cif", "rb").read(), "structure.cif")
print(summary(p)["band_gap_eV"], p.bands.shape)   # bands: (k points, bands), eV relative to the VBM or E_F
```

Lower level, any **k** points, with `epnet.models.general_epm.GeneralEPM.from_checkpoint(torch.load(...))` and its
`predict_bands(structure, kpoints, n_bands)`.

**Web application (local).** `pip install -e .[app]`, then `python -m apps.epnet_bands.app --port 7860` and open
<http://127.0.0.1:7860>. See `apps/epnet_bands/README.md` for a server deployment.

## Models

`models/epnet_spd_5cN_s0.pt` and `models/epnet_spd_5cN_s1.pt` are the final model (two independently trained seeds;
the paper reports their mean; the app uses seed 0). `models/stages/` holds every checkpoint of the training chain,
starting from the *s*/*p* model EpNet-sp (`epnet_sp_g4_s0.pt`); see `models/README.md`.

## Reproduce the paper

The commands run from the repository root. Generated files (band data, runs, scores) go to `data/`, `runs/` and
`results/`; `.gitignore` lists them.

**1. Data.** The paper's manifest is included: `data/manifests/general_spd_v1.json`, 22,553 Materials Project
structures of up to 24 sites with their splits (by chemical system, then structure type), the valence contract (semicore
bands removed) and +U and spin-polarized calculations excluded; `data/manifests/bigcell_spd_v1.json` and
`bigcell_sp_v1.json` hold the 25-60-site test cells, and `data/splits/general_spd_v1_splits.csv.gz` the split of every
structure as a table. The band structures themselves are downloaded from the Materials Project (an API key in
`MP_API_KEY`; `pip install -e .[data]`) into `data/mp_bands/`, where the manifests point:

    python scripts/download_mp_bands.py --out data/mp_bands --max-nsites 60     # 60: also the 25-60-site test cells

The paper's data were downloaded in September 2026 (79,081 materials with band structures); later database releases
may differ. A manifest for new data is built with `scripts/build_general_manifest.py --scope spd --max-sites 24
--no-holdout-elements --output data/manifests/<name>.json`; the paper's also kept the splits of the earlier *s*/*p*
manifest (`--inherit`), so a rebuilt one assigns some structures differently.

**2. The *d* channels** (`epnet/data/atomic_channels/`, included) were built from scalar-relativistic all-electron
atoms in five steps, each checked against gates recorded in `results/checks/` (the paper's reports are included):

    python scripts/check_atomic_channels.py        # pseudo-orbitals (all-electron tail kept) and their checks
    python scripts/check_ae_local.py               # the all-electron-tail local potential
    python scripts/build_two_reference.py          # the second d reference energy (eps_d + 4 eV)
    python scripts/safeguard_two_reference_v2.py   # per-element offset (eps_d + 3 eV; Sc + 6 eV) and stretch bound
    python scripts/build_two_reference.py          # the second reference rebuilt at the chosen offsets

The steps overwrite the tables and reports in place. `results/checks/atomic_channels_report.json` is the
norm-conserving comparison (`check_atomic_channels.py --renormalized`); the construction used is the default (ultrasoft).

**3. Training.** Stages run in sequence, each from the previous one; `configs/stages/` holds the exact arguments of every
run of the paper, and `scripts/train_stage.py` turns them into the trainer's command line:

| Stage | Cells | Steps | Learning rate | GPUs | Hours | Starts from |
|---|---|---|---|---|---|---|
| J-A2 (seed 1) | *d*¹⁰ compounds, ≤ 3 sites; *s*/*p* anchors of 4-8 sites | 3,000 | 1e-3 | 1 | 6.4 | EpNet-sp |
| 4a-A (0), 4b-S (1), 4c-S (0) | early *d*, 4*d*/5*d*, open-shell 3*d*, ≤ 3 sites; the same anchors | 3,000 each | 1e-3 | 1 | 5-7.5 each | the previous stage |
| 5a (seeds 0, 1) | all cells ≤ 8 sites | 6,000 | 1e-3 | 1 | 15 | 4c-S |
| 5aL (seeds 0, 1) | the same | 16,000 | 1e-4 | 4 | 20 | 5a |
| 5cL (seeds 0, 1) | all 12,136 training cells ≤ 24 sites, new cells above 8 sites weighted 4 | 11,500 | 1e-4 | 4 | | 5aL |
| 5cN (seeds 0, 1) | the same, *s*/*p* cells weighted 1.5 | 22,000 | 5e-5 | 4 | 17-18 | 5cL |

    python scripts/train_stage.py configs/stages/5cN_s0.json --dry-run       # the command (torchrun with 4 ranks)
    python scripts/train_stage.py configs/stages/5cN_s0.json                 # run it

The configs record the number of GPUs of each run ("ranks"). With several ranks the training cells are sharded and each
rank draws its share of every batch from its own cells, so an exact replay uses the recorded number (`--ranks` changes
it). The hours are those of the paper's runs (5cL on L40S, 5cN on H100 GPUs) and vary with the GPU model.
Some runs
record a wall-clock limit of their cluster jobs (`max_hours`); a run that reaches it saves its state and stops, and
the same command resumes it (the configs set `--resume`).

**4. Evaluation.** Score the held-out sets with the final model (per-structure results of the paper:
`results/stage5/5cN_scores_s{0,1}.csv.gz`):

    mkdir -p results/stage5/5cN/s0 && cp models/epnet_spd_5cN_s0.pt results/stage5/5cN/s0/best_smoothed.pt
    python scripts/stage5_eval.py --arm 5cN --run 0 --device cuda       # held-out chemical systems, structure types, ...
    python scripts/stage5_eval.py --arm 5cN --run 0 --device cuda --cells-file results/stage5/closeout/cells.json --out-prefix closeout_chunk_

**5. External test on Alexandria.** Work files go to `$EPNET_ALEXANDRIA_DIR` (default `data/alexandria`); the
download uses `curl`:

    for k in $(seq 0 57); do python results/alexandria/fetch_filter.py $k; done    # PBE release 2025.07.02, 58 files
    python results/alexandria/dedupe_mp.py                  # remove structures with a Materials Project twin
    python results/alexandria/make_sample.py                # the 5,000-structure sample of the comparison
    python results/alexandria/score_alexandria.py --sample $EPNET_ALEXANDRIA_DIR/sample.json \
        --ckpt models/epnet_spd_5cN_s0.pt --out results/alexandria/5cN_s0/score_part0.json   # full mode, on the sample
    python results/alexandria/report.py 5cN_s0              # its report
    python results/alexandria/screen/prepare_screen.py      # the full screen, in 400 parts
    python results/alexandria/screen/screen_alexandria.py --ckpt models/epnet_spd_5cN_s0.pt --part 0   # ... --part 399
    python results/alexandria/screen/report_screen.py       # summary.json, parity_hist.json (the paper's are included)

## Repository layout

    epnet/                  model, Hamiltonian, solvers, data handling and training (Python package)
    apps/epnet_bands/       prediction core and web application
    models/                 final model and the checkpoints of the training chain
    configs/                recorded training arguments and the cell lists they use
    scripts/                prediction, data, d channels, training, evaluation
    data/splits/            split of every structure of the training manifest
    results/                evaluation inputs, per-structure scores, Alexandria external test
    tests/                  checks of the models, the configs and predictions

## Citation

Please cite the paper (reference to be added on publication) and, for the environment encoder,
Q. Zhu, *EpNet-Si: Atomic Form Factors Learned from Band Structures*, arXiv:2610.14992 (2026).

## License

MIT (see `LICENSE`). Training labels are Materials Project data (CC BY 4.0); the external test uses the Alexandria
database.

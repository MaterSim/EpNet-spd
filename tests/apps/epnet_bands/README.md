# EpNet band-structure web application

Upload a crystal structure and get its band structure and density of states: an interactive plot and downloads
(bands CSV, DOS CSV, PNG, JSON summary, standardized cell). A public instance runs at <https://mmi.charlotte.edu/epnet>.

## Run locally

    pip install -e ".[app]"
    python -m apps.epnet_bands.app --port 7860           # http://127.0.0.1:7860

Settings (environment variables):
- `EPNET_THREADS`: CPU threads for predictions (default: all cores).
- `EPNET_KSPLIT`: worker processes sharing the k points of a prediction, each with EPNET_THREADS / EPNET_KSPLIT
  threads (default: one per 4 threads; 1 turns it off). On 16 cores this is 1.7-9x faster than one process.
- `EPNET_CKPT`: the model (default `models/epnet_spd_5cN_s0.pt`).
- `EPNET_APP_PASSWORD_FILE`: a file with `user:password` to require a login.
- `GRADIO_ROOT_PATH`: the URL sub-path when served behind a reverse proxy under one (e.g. `/epnet`).

Options: `--host` (default 127.0.0.1; keep it behind a reverse proxy such as nginx, with websocket upgrade headers),
`--port`. `--backend slurm` runs each prediction as an `srun` step (`--partition`, `--gpu-partition`) for HPC
clusters where the web process and the compute nodes are separate.

## Files

- `core.py`: one prediction (standardized primitive cell, Setyawan-Curtarolo path, bands; DOS on the symmetry-reduced
  mesh; warnings for open-shell 3*d* elements, +U oxides and fluorides, cells larger than the training cells) and the
  downloads. Uploads are parsed as structures only.
- `app.py`: the page (Gradio); `worker.py`: the Slurm step; `element_counts.json`: training structures per element of
  the final model, for the periodic table on the page; `examples/`: Materials Project cells.

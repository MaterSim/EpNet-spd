# EpNet models

Each file is a `GeneralEPM` checkpoint (`epnet/models/general_epm.py`): the model class, its species, its
configuration and its weights (no optimizer state). Load with

```python
import torch
from epnet.models.general_epm import GeneralEPM
model = GeneralEPM.from_checkpoint(torch.load("models/epnet_spd_5cN_s0.pt", map_location="cpu", weights_only=False)).eval()
```

## Final model

| File | |
|---|---|
| `epnet_spd_5cN_s0.pt` | final model, seed 0 (used by the web application and `scripts/predict.py`) |
| `epnet_spd_5cN_s1.pt` | final model, seed 1 (trained independently; the paper reports the mean of the two) |

63 elements (34 *s*/*p*, 29 *d*), 78,964 parameters; plane-wave cutoff 12 Ry, 16 Ry with B or C, 20 Ry with N, O, F or
a *d* element. Trained on 12,136 Materials Project PBE band structures of up to 24 sites (non-magnetic, no +U).
Scope and accuracy: see the top-level README.

## Training chain (`stages/`)

Every checkpoint that the next stage started from, so that any stage can be rerun on its own with
`scripts/train_stage.py configs/stages/<stage>_s<seed>.json`:

| File | Stage | Elements | Starts from |
|---|---|---|---|
| `epnet_sp_g4_s0.pt` | EpNet-sp, the *s*/*p* model | 34 | |
| `J-A2_s1.pt` | *d*¹⁰ elements added, cells ≤ 3 sites | 40 | `epnet_sp_g4_s0.pt` |
| `4a-A_s0.pt` | early *d* | 48 | `J-A2_s1.pt` |
| `4b-S_s1.pt` | 4*d*/5*d* | 58 | `4a-A_s0.pt` |
| `4c-S_s0.pt` | open-shell 3*d* | 63 | `4b-S_s1.pt` |
| `5a_s0.pt`, `5a_s1.pt` | all cells ≤ 8 sites | 63 | `4c-S_s0.pt` |
| `5aL_s0.pt`, `5aL_s1.pt` | the same, continued at lower learning rate | 63 | `5a_s{0,1}.pt` |
| `5cL_s0.pt`, `5cL_s1.pt` | all cells ≤ 24 sites | 63 | `5aL_s{0,1}.pt` |
| (final) `../epnet_spd_5cN_s{0,1}.pt` | final stage | 63 | `5cL_s{0,1}.pt` |

Selected checkpoints are the run's `best_smoothed.pt`: the checkpoint saved at the evaluation whose selection-set median
band error, averaged over three consecutive evaluations (centred), is lowest. Where the next stage continued from the end
of a run (`5aL_s0`, `5cL_s0`, `5cL_s1`), the last checkpoint is published instead.

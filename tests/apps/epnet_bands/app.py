"""EpNet band-structure web app (Gradio; private, cluster-hosted; 2026-10-16 for EpNet-sp, s/p/d model 2026-11-04).

  EPNET_CKPT=<checkpoint>  EPNET_APP_PASSWORD_FILE=<file with "user:password">  python -m apps.epnet_bands.app --port 7860

Upload a CIF / POSCAR (or anything ASE reads): the app returns an interactive band plot and downloads (CSV of the
bands along the path, PNG, JSON summary, the standardized primitive cell as POSCAR). No share link is created.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import subprocess
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .core import MAX_SITES, WARN_SITES, figure_png, load_model, path_breaks, predict, summary, to_csv, to_dos_csv

HERE = Path(__file__).resolve().parent
OCC, EMPTY, GRID = "#1f8a8a", "#eb6834", "#d9d8d2"

# periodic table of the supported elements, as manuscript Fig. 1e (user, 2026-11-04): shaded by the number of training
# structures that contain each element (log scale 30-3,000, the figure's blue sequential map); outlined: outside the scope
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
       "#184f95", "#104281", "#0d366b"]
PT_ROWS = [
    ["H"] + [""] * 16 + ["He"],
    ["Li", "Be"] + [""] * 10 + ["B", "C", "N", "O", "F", "Ne"],
    ["Na", "Mg"] + [""] * 10 + ["Al", "Si", "P", "S", "Cl", "Ar"],
    ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"],
    ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"],
    ["Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"],
]


def _seq(t: float) -> str:
    """Colour at t in [0, 1] on SEQ (linear in RGB between the stops)."""
    t = min(max(t, 0.0), 1.0) * (len(SEQ) - 1)
    i = min(int(t), len(SEQ) - 2)
    a, b = (np.array([int(c[k:k + 2], 16) for k in (1, 3, 5)], float) for c in (SEQ[i], SEQ[i + 1]))
    r, g, bl = (a + (t - i) * (b - a)).round().astype(int)
    return f"#{r:02x}{g:02x}{bl:02x}"


def periodic_table_html(elements, counts: dict) -> str:
    lo, hi = math.log(30), math.log(3000)
    cells = []
    for r, row in enumerate(PT_ROWS):
        for c, el in enumerate(row):
            if not el:
                continue
            pos = f"grid-row:{r + 1};grid-column:{c + 1};"
            if el in elements:
                n = int(counts.get(el, 0))
                col = _seq((math.log(max(n, 30)) - lo) / (hi - lo))
                lum = sum(w * int(col[k:k + 2], 16) / 255 for w, k in ((0.299, 1), (0.587, 3), (0.114, 5)))
                tip = f"{el}: {n:,} training structures" if n else f"{el}: supported"
                cells.append(f'<div class="ept-c" title="{tip}" style="{pos}background:{col};color:'
                             f'{"#ffffff" if lum < 0.55 else "#0b0b0b"};">{el}</div>')
            else:
                cells.append(f'<div class="ept-c ept-x" title="{el}: outside the scope" style="{pos}">{el}</div>')
    ticks = "".join(f'<span style="left:{100 * (math.log(v) - lo) / (hi - lo):.1f}%">{v:,}</span>' for v in (30, 100, 300, 1000, 3000))
    grad = ",".join(SEQ)
    legend = (f'<div class="ept-leg" style="grid-row:1 / span 3;grid-column:3 / span 10;">'
              f'<div class="ept-lt">training structures per element</div>'
              f'<div class="ept-bar" style="background:linear-gradient(to right,{grad});"></div>'
              f'<div class="ept-ticks">{ticks}</div>'
              f'<div class="ept-note"><span class="ept-sw"></span>outside the scope · {len(elements)} elements supported</div></div>')
    css = ("<style>.ept{display:grid;grid-template-columns:repeat(18,minmax(0,1fr));gap:3px;max-width:760px;margin:6px 0 4px;"
           "font-family:system-ui,sans-serif}"
           ".ept-c{aspect-ratio:1/0.9;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;"
           "border-radius:3px;cursor:default}"
           ".ept-x{background:transparent;color:rgba(128,128,128,.85);font-weight:400;"
           "box-shadow:inset 0 0 0 1px rgba(128,128,128,.45)}"
           ".ept-leg{align-self:center;padding:0 22px 0 14px;font-size:11px}"
           ".ept-lt{margin-bottom:3px;opacity:.85}.ept-bar{height:10px;border-radius:2px}"
           ".ept-ticks{position:relative;height:14px}.ept-ticks span{position:absolute;transform:translateX(-50%);top:1px}"
           ".ept-note{margin-top:4px;opacity:.85;display:flex;align-items:center;gap:5px}"
           ".ept-sw{display:inline-block;width:12px;height:11px;border-radius:2px;box-shadow:inset 0 0 0 1px rgba(128,128,128,.45)}"
           "</style>")
    return css + '<div class="ept">' + "".join(cells) + legend + "</div>"



def plot(p, emin, emax):
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.76, 0.24], horizontal_spacing=0.02)
    sel = (p.dos_e >= emin - 0.5) & (p.dos_e <= emax + 0.5)
    fig.add_trace(go.Scatter(x=p.dos[sel], y=p.dos_e[sel], mode="lines", fill="tozerox", line=dict(color=OCC, width=1.4),
                             fillcolor="rgba(31,138,138,0.22)", showlegend=False, customdata=p.dos_int[sel],
                             hovertemplate="E = %{y:.3f} eV<br>DOS %{x:.3f} /eV<br>N(E) %{customdata:.3f} e<extra></extra>"), row=1, col=2)
    if p.dos_top < emax:
        fig.add_hrect(y0=p.dos_top, y1=emax, fillcolor="#eeeeea", line_width=0, row=1, col=2,
                      annotation_text="beyond computed bands", annotation_font_size=10)
    cut = path_breaks(p)                              # branch jumps: a gap, not a vertical line
    xs = np.insert(np.asarray(p.distance, float), cut, np.nan)
    labs = np.insert(np.array(p.labels, dtype=object), cut, "")
    for b in range(p.bands.shape[1]):
        occ = p.metal or b < p.n_occupied
        fig.add_trace(go.Scatter(x=xs, y=np.insert(p.bands[:, b], cut, np.nan), mode="lines", connectgaps=False,
                                 line=dict(color=OCC if occ else EMPTY, width=1.6),
                                 name=f"band {b + 1}", showlegend=False,
                                 customdata=labs,
                                 hovertemplate=f"band {b + 1}<br>%{{customdata}}<br>E = %{{y:.3f}} eV<extra></extra>"), row=1, col=1)
    for x, _ in p.ticks:
        fig.add_vline(x=x, line=dict(color=GRID, width=1), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="#6b6a64", width=1, dash="dot"))
    ref = "E<sub>VBM</sub>" if p.reference == "VBM" else "E<sub>F</sub>"
    gap = "metal" if p.metal else f"gap {p.gap:.3f} eV ({'direct' if p.gap_direct else 'indirect'})"
    fig.update_layout(title=f"{p.formula} · {p.spacegroup} · {gap}", template="simple_white", height=520,
                      margin=dict(l=60, r=20, t=60, b=40),
                      xaxis=dict(tickmode="array", tickvals=[t[0] for t in p.ticks], ticktext=[t[1] for t in p.ticks],
                                 range=[p.distance[0], p.distance[-1]], showgrid=False),
                      xaxis2=dict(title="DOS (states/eV)", rangemode="tozero", showgrid=False),
                      yaxis=dict(title=f"E − {ref} (eV)", range=[emin, emax], zeroline=False))
    return fig


BACKEND = {"mode": "local"}
JOBS = Path(os.environ.get("EPNET_APP_JOBS", Path.home() / ".epnet_app_jobs"))


def _srun_cmd(job: Path, gpu: bool) -> list[str]:
    """One srun step for the worker: a GPU if one frees up within --gpu-wait seconds (2026-11-04: MoS2 30 s on an A100
    against about 2 min on 16 CPU cores), else 16 cores on the CPU partition."""
    if gpu:
        res = ["--partition", BACKEND["gpu_partition"], "--gres", "gpu:1", f"--immediate={BACKEND['gpu_wait']}",
               "--cpus-per-task", "8"]
        threads = 8
    else:
        res = ["--partition", BACKEND["partition"], *(["--exclude", BACKEND["exclude"]] if BACKEND["exclude"] else []), "--cpus-per-task", "16"]
        threads = 16
    return ["srun", *res, "--mem", "24G", "--time", "00:20:00", "--job-name", "epnet-bands-req",
            f"--export=ALL,OMP_NUM_THREADS={threads},EPNET_CKPT={os.environ['EPNET_CKPT']}",
            str(BACKEND["python"]), "-W", "ignore", "-m", "apps.epnet_bands.worker", str(job)]


def compute(data: bytes, name: str, density: float, n_empty: int, sigma: float = 0.1):
    """local: in this process; slurm: one short srun step on a compute node (login node keeps only the UI)."""
    if BACKEND["mode"] == "local":
        try:
            return predict(data, name, line_density=density, n_empty=n_empty, dos_sigma=sigma)
        except ValueError as ex:
            raise gr.Error(str(ex))
    JOBS.mkdir(mode=0o700, exist_ok=True)
    job = Path(tempfile.mkdtemp(prefix="job_", dir=JOBS))
    (job / "input.bin").write_bytes(data)
    (job / "params.json").write_text(json.dumps({"filename": name, "line_density": density, "n_empty": n_empty, "dos_sigma": sigma}))
    r = None
    for gpu in ([True, False] if BACKEND["gpu_partition"] else [False]):
        r = subprocess.run(_srun_cmd(job, gpu), cwd=BACKEND["root"], capture_output=True, text=True, timeout=1500)
        if (job / "error.txt").exists():
            raise gr.Error((job / "error.txt").read_text())
        if (job / "result.pkl").exists():
            break
        # no GPU within the wait (srun --immediate), or the GPU step failed: the same job on CPU
    if not (job / "result.pkl").exists():
        raise gr.Error("the compute job failed: " + (r.stderr.strip().splitlines() or ["no output"])[-1][:300])
    return pickle.loads((job / "result.pkl").read_bytes())


def run(upload, text, density, n_empty, emin, emax, sigma=0.1):
    if upload is None and not (text or "").strip():
        raise gr.Error("Upload a structure file or paste one.")
    if upload is not None:
        path = Path(upload if isinstance(upload, str) else upload.name)
        data, name = path.read_bytes(), path.name
    else:
        data, name = text.encode(), "pasted.cif" if "_cell_length_a" in text else "POSCAR"
    p = compute(data, name, float(density), int(n_empty), float(sigma))
    out = Path(tempfile.mkdtemp(prefix="epnet_out_"))
    stem = f"{p.formula}_epnet"
    (out / f"{stem}_bands.csv").write_text(to_csv(p))
    (out / f"{stem}_dos.csv").write_text(to_dos_csv(p))
    (out / f"{stem}_bands.png").write_bytes(figure_png(p, float(emin), float(emax)))
    (out / f"{stem}_summary.json").write_text(json.dumps(summary(p), indent=1))
    (out / f"{stem}_structure.vasp").write_text(p.poscar)
    gap = ("**Metal** (energies relative to E_F)" if p.metal else
           f"**Gap {p.gap:.3f} eV**, {'direct' if p.gap_direct else 'indirect'} (VBM at {p.vbm_k}, CBM at {p.cbm_k}); energies relative to the VBM")
    md = (f"### {p.formula} · {p.spacegroup}\n{gap}\n\n"
          f"{p.nsites} sites in the standardized primitive cell · {p.n_electrons} valence electrons ({p.n_occupied} occupied bands) · "
          f"{len(p.distance)} path k-points · DOS on a {'×'.join(map(str, p.dos_mesh))} mesh (σ {p.dos_sigma} eV) · "
          f"{p.timing_s:.1f} s on {p.model['device']}\n\n"
          + "".join(f"- {w}\n" for w in p.warnings))
    files = [str(out / f"{stem}_{x}") for x in ("bands.csv", "dos.csv", "bands.png", "summary.json", "structure.vasp")]
    return plot(p, float(emin), float(emax)), md, files


_counts_file = HERE / "element_counts.json"          # element_counts.py (training split of the deployed model)
ELEMENT_COUNTS = json.loads(_counts_file.read_text())["counts"] if _counts_file.exists() else {}


def build():
    info = load_model()[2] if BACKEND["mode"] == "local" else _info_only()
    about = (f"**EpNet** predicts PBE-like band structures with a learned empirical pseudopotential Hamiltonian trained on "
             f"Materials Project PBE band energies. Scope: non-magnetic, no spin–orbit coupling, no +U, valence bands only, "
             f"63 elements of the s, p and d blocks (periodic table below, as Fig. 1e of the paper). Trained on cells of up to {WARN_SITES} sites "
             f"(limit here {MAX_SITES}). Bands are computed on the standardized primitive cell along the Setyawan–Curtarolo path. "
             f"Accuracy on held-out chemical systems: median band error 0.061 eV near the gap or E_F and median gap error "
             f"0.102 eV; on 600,617 Alexandria structures absent from the Materials Project, median gap error 0.108 eV and "
             f"98.9% correctly metal or insulator. Least accurate: hydrides and first-row elements, layered Mo/W dichalcogenide "
             f"slabs, and dense high-coordination polymorphs unlike any training structure. Model: `{Path(info['checkpoint']).name}` "
             f"(final model 5c-N, seed 0).")
    with gr.Blocks(title="EpNet band structures", analytics_enabled=False) as demo:
        gr.Markdown("# EpNet band structure\n" + about)
        gr.HTML(periodic_table_html(set(info["elements"]), ELEMENT_COUNTS))
        with gr.Row():
            with gr.Column(scale=1):
                up = gr.File(label="Structure file (CIF, POSCAR, …)", file_count="single", type="filepath")
                txt = gr.Textbox(label="…or paste CIF / POSCAR text", lines=6)
                with gr.Accordion("Options", open=False):
                    dens = gr.Slider(10, 60, value=25, step=5, label="k-points per Å⁻¹ along the path")
                    nemp = gr.Slider(2, 20, value=8, step=1, label="empty bands (minimum; more are added for d atoms and larger cells)")
                    sig = gr.Slider(0.02, 0.3, value=0.1, step=0.01, label="DOS Gaussian broadening (eV)")
                    emin = gr.Number(value=-6.0, label="plot E min (eV)")
                    emax = gr.Number(value=6.0, label="plot E max (eV)")
                btn = gr.Button("Predict band structure", variant="primary")
                ex = sorted(str(p) for p in (HERE / "examples").glob("*.cif"))
                if ex:
                    gr.Examples(examples=[[e] for e in ex], inputs=[up], label="Examples (Materials Project cells)")
            with gr.Column(scale=2):
                fig = gr.Plot(label="Band structure")
                md = gr.Markdown()
                files = gr.Files(label="Downloads: bands CSV, DOS CSV, PNG, JSON summary, standardized cell (POSCAR)")
        btn.click(run, [up, txt, dens, nemp, emin, emax, sig], [fig, md, files], concurrency_limit=1)
    return demo


def _info_only():
    """Model metadata without loading torch weights on the login node."""
    import torch
    ck = torch.load(os.environ["EPNET_CKPT"], map_location="cpu", weights_only=False)
    return {"checkpoint": os.environ["EPNET_CKPT"], "elements": list(ck["species_symbols"]), "device": "cpu (Slurm)"}


def main():
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--backend", choices=("local", "slurm"), default="local")
    ap.add_argument("--partition", default="", help="Slurm backend: CPU partition")
    ap.add_argument("--exclude", default="", help="Slurm backend: compute nodes to avoid")
    ap.add_argument("--gpu-partition", default="", help="Slurm backend: try a GPU on this partition first (default: CPU only)")
    ap.add_argument("--gpu-wait", type=int, default=60, help="seconds to wait for a free GPU before using the CPUs")
    a = ap.parse_args()
    BACKEND.update(mode=a.backend, partition=a.partition, exclude=a.exclude, gpu_partition=a.gpu_partition, gpu_wait=a.gpu_wait, python=sys.executable, root=str(Path(__file__).resolve().parents[2]))
    os.environ.setdefault("EPNET_CKPT", str(HERE.parents[1] / "models/epnet_spd_5cN_s0.pt"))
    auth = None
    pw = os.environ.get("EPNET_APP_PASSWORD_FILE")
    if pw:
        user, _, password = Path(pw).read_text().strip().partition(":")
        auth = [(user, password)]
    if a.backend == "local":     # CPU server: k-point parallel workers, one per 4 threads unless EPNET_KSPLIT is set
        threads = int(os.environ.get("EPNET_THREADS") or os.cpu_count() or 1)
        split = int(os.environ.get("EPNET_KSPLIT") or max(1, threads // 4))
        if split > 1:
            from .core import start_pool
            start_pool(split, max(1, threads // split))
            print(f"k-point parallel: {split} worker processes x {max(1, threads // split)} threads", flush=True)
    build().queue(max_size=8).launch(server_name=a.host, server_port=a.port, share=False, auth=auth, show_api=False,
                                     max_file_size="2mb")


if __name__ == "__main__":
    main()

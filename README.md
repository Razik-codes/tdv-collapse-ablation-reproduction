# TDV collapse ablation — independent reproduction

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/Razik-codes/tdv-collapse-ablation-reproduction/blob/main/claim_tutorial.py)
[![Quality checks](https://github.com/Razik-codes/tdv-collapse-ablation-reproduction/actions/workflows/quality.yml/badge.svg)](https://github.com/Razik-codes/tdv-collapse-ablation-reproduction/actions/workflows/quality.yml)
[![Licensing: see notices](https://img.shields.io/badge/Licensing-see%20notices-6b7280.svg)](THIRD_PARTY_NOTICES.md)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](claim_tutorial.py)

> **Independent research artifact by [Razik-codes](https://github.com/Razik-codes).** This project is not authored by, affiliated with, or endorsed by the TDV paper authors. It studies one TDV claim at reduced scale and cites the original work throughout.

## My contribution

I designed the low-rank-motion synthetic proxy, implemented the downscaled control and two ablations, ran the experiments on a 4 GB GPU, built the interactive tutorial, and wrote the analysis and limitations. The vendored TDV training and evaluation stack remains upstream work; see [CONTRIBUTIONS.md](CONTRIBUTIONS.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

## Research note


I started from the TDV codebase and investigated whether the collapse reported in Table 4 could be observed on a single 4 GB GPU. I first tried a `testsrc2` synthetic source, then replaced it with clips containing a moving textured disc over a mostly static background after finding that the original source did not provide the low-rank motion signal assumed by TDV. I ran the full model and two ablations with the same short training command, changing one component at a time.

Because I did not have the resources to reproduce full SSv2 pretraining and the ImageNet KNN evaluation, I used next-frame prediction gain over the identity baseline as a smaller diagnostic. The result below is therefore a **qualitative reproduction of the mechanism**, not a claim that I recovered the paper's full-scale numbers. The [full report](reports/collapse-ablation/report.md) records the substitutions, negative control, compute, and limitations.

The original TDV model and evaluation stack are documented below this study. See [third-party and source provenance](THIRD_PARTY_NOTICES.md) for the relationship between the original code and my additions.

To inspect the tutorial locally with its declared Python 3.11+ dependencies, install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv run claim_tutorial.py
```

**Claim (paper Table 4).** TDV's two load-bearing components — the **motion encoder** and the **MSE next-frame-prediction loss** — are each *critical*: remove either and training collapses (ImageNet-KNN Top-5 falls from **17.05%** to near-chance **1.87% / 1.58%**).

**What I ran.** I downscaled the setup to fit a single **RTX 3050 Ti (4 GB)**: ViT-S/14 from scratch, batch 1, 600 steps, on synthetic **low-rank-motion** clips (a textured disc on a static background — faithful to TDV's assumption). Without ImageNet, I used a direct read on TDV's own objective instead of the KNN collapse detector — the **next-frame prediction gain over the identity baseline** (`ẑ_{t+1}=z_t+Δz_t` vs `z_t`). Every arm used the *identical* command; each differed by exactly one committed knob.

**Verdict: ✅ reproduced (qualitatively).** Only the full recipe learns to predict the next frame; removing either component breaks it — the same ordering as Table 4, with a large margin:

| Arm | Paper (ImageNet KNN Top-5) | This reproduction (prediction gain vs identity) |
|---|---:|---:|
| **Full TDV** | **17.05%** ✅ | **+42.6%** ✅ |
| − Motion encoder | 1.87% ❌ | **0.0%** ❌ |
| − MSE loss | 1.58% ❌ | **−447%** ❌ |

The paper's exact KNN numbers are **not** reproducible here (no ImageNet / no full-scale SSv2 pretraining) and are not claimed. **Compute:** the 3 reproduction runs ≈ 21 min total on one 4 GB laptop GPU.

- 📓 **Interactive tutorial:** [open in molab](https://molab.marimo.io/github/Razik-codes/tdv-collapse-ablation-reproduction/blob/main/claim_tutorial.py) · source [`claim_tutorial.py`](claim_tutorial.py) — frozen results render instantly; an optional GPU lab (DINOv2 ViT-S Δz sweep, behind a run button) is designed for molab's RTX PRO 6000.
- 📄 **Full write-up:** [`reports/collapse-ablation/report.md`](reports/collapse-ablation/report.md) — method, exact substitutions, figures, lineage, and honest limitations.

### Re-run the reproduction

Create the Python 3.11 environment described in [Setup](#setup), ensure `ffmpeg` is available, then run each arm from the repository root. `PYTHON_BIN` may point to any configured interpreter; generated videos are written to the ignored `smoke_data/` directory.

```bash
PYTHON_BIN=python TDV_ARM=full bash job_scripts/pretrain_tdv_local_smoketest.sh
PYTHON_BIN=python TDV_ARM=no-motion bash job_scripts/pretrain_tdv_local_smoketest.sh
PYTHON_BIN=python TDV_ARM=no-mse bash job_scripts/pretrain_tdv_local_smoketest.sh
```

These commands reproduce the downscaled mechanism-level study, not the paper-scale SSv2/ImageNet evaluation.

### Experiment log

Every arm uses the same public harness; `TDV_ARM` selects the one controlled ablation. The public harness consolidates the three recorded configurations behind a single explicit selector. Round 2 uses faithful low-rank-motion data; round 1 (`testsrc2`) is kept as the negative control that motivated it.

| Branch / experiment | Purpose / change vs control | Exact run command | Outcome | Compute |
|---|---|---|---|---|
| `main` | Public three-arm reproduction harness | `TDV_ARM={full,no-motion,no-mse} bash job_scripts/pretrain_tdv_local_smoketest.sh` | reruns the downscaled arms | 1× 4 GB GPU |
| Full TDV control | **Full TDV control** (round 2, faithful data) | `TDV_ARM=full bash job_scripts/pretrain_tdv_local_smoketest.sh` | **+42.6%** gain — healthy ✅ | RTX 3050 Ti, ~7 min |
| − Motion encoder | `+ --remove_motion_encoder` | `TDV_ARM=no-motion bash job_scripts/pretrain_tdv_local_smoketest.sh` | **0.0%** gain — fails ❌ | RTX 3050 Ti, ~7 min |
| − MSE loss | remove `--use_mse_loss` | `TDV_ARM=no-mse bash job_scripts/pretrain_tdv_local_smoketest.sh` | **−447%** gain — collapse ❌ | RTX 3050 Ti, ~7 min |
| Round-1 negative control | `testsrc2` data with collapse metrics | `TDV_ARM=full bash job_scripts/pretrain_tdv_local_smoketest.sh` | Confounded → motivated faithful data | RTX 3050 Ti, ~8 min |

---

# Upstream TDV reference code

> The documentation below describes the vendored upstream TDV project and paper for reference. It is not a claim that this artifact was authored by the TDV authors or that it reproduces their full-scale benchmark results.

## You Don't Need Strong Assumptions: Temporal Difference Visual Representation Learning

📚 [Paper](https://temporal-difference-vision.github.io/static/pdfs/tdv.pdf) | 🌐 [Website](https://temporal-difference-vision.github.io/) | 🧾 [Bibtex](#citations)

**Ninad Daithankar\*, Alexi Gladstone\*, Yann LeCun, Heng Ji** &nbsp;(\*Equal Contribution)  
University of Illinois Urbana-Champaign &nbsp;·&nbsp; New York University

<img src="assets/intuition.gif" width="75%">


TDV is a self-supervised video representation learning method built on a single causal assumption: **the past causes the future**. A frame encoder and motion encoder are trained jointly such that the current frame's representation plus the encoded motion equals the next frame's representation — no augmentations, masking, or cropping required. TDV matches DINO, iBOT baselines on segmentation tasks and surpasses them on optical flow and stereo depth.


## Setup

Create a conda environment and install dependencies:

```bash
# Create the environment
conda create -n tdv python=3.11
conda activate tdv

# PyTorch (CUDA 12.6) — works on both x86_64 and aarch64/GH200
pip install torch==2.10.0+cu126 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu126 

# aiohttp==3.8.1 can't compile from source on Python 3.11; conda provides a compatible binary
conda install -c conda-forge aiohttp libiconv ffmpeg -y  

# --no-deps avoids conflicts while downloading dependencies
pip install -r requirements.txt --no-deps  
```

Set environment variables:

```bash
export HF_HOME=/path/to/cache
export HF_TOKEN=your_token_here  # may not be needed
```

Login to wandb:

```bash
wandb login
```

> **WandB Tip:** in the wandb UI, switch the x-axis from the default *Step* to `trainer/global_step` to see correct plots — the default step counter does not account for gradient accumulation.

For dataset setup see [data/cv/README.md](data/cv/README.md).

## Running the Code

### Training

#### Running locally

In [job_scripts/pretrain_tdv.slurm](job_scripts/pretrain_tdv.slurm), set:

| Field | Location | What to set |
|---|---|---|
| `RUN_NAME` | `export RUN_NAME=` | name for this run (used in logs and wandb) |
| `--aggr_dataset_dirs` | `ARGS` block | path(s) to your dataset(s) |
| `--wandb_project` | `ARGS` block | your wandb project name |
| `--gpus` | `ARGS` block | e.g. `"[0]"` for single GPU, `"[0, 1]"` for two (default is -1 which uses all GPUs) |
| `--knn_eval_data_dir` | `ARGS` block | path to your ImageNet folder if downloaded elsewhere |

Then run:

```bash
bash job_scripts/pretrain_tdv.slurm
```

For debugging, also add `--debug_mode` to the `ARGS` block — it disables wandb, enables anomaly detection, and limits batches. Or add `--no_wandb` to skip wandb without the other debug constraints.

#### Running on a cluster (SLURM)

**1. Set run-specific fields** in [job_scripts/pretrain_tdv.slurm](job_scripts/pretrain_tdv.slurm) — same as above (`RUN_NAME`, `--aggr_dataset_dirs`, `--wandb_project`, `--knn_eval_data_dir`). 

Then, update the `#SBATCH` header lines at the top of the file:

| Field | What to set |
|---|---|
| `#SBATCH --job-name` | same as `RUN_NAME` |
| `#SBATCH --output` | log path, e.g. `logs/slurm/YOUR_RUN_NAME` |
| `#SBATCH --nodes` | number of nodes |
| `#SBATCH --gpus-per-node` | GPUs per node |
| `#SBATCH --time` | wall time limit |

**2. Set cluster-specific settings** in your cluster's header file under [job_scripts/slurm_headers/](job_scripts/slurm_headers/) (copy `example_gh200.slurm` to `<cluster_name>.slurm`):

| Field | What to set |
|---|---|
| `#SBATCH --account` | your cluster account/allocation |
| `#SBATCH --partition` | partition name for your cluster |

**3. Submit:**

Make sure the `tdv` conda environment is activated before submitting — SLURM inherits it from the shell that calls `sbatch`.

```bash
conda activate tdv
bash slurm_executor.sh <cluster_name> job_scripts/pretrain_tdv.slurm
```

`slurm_executor.sh` prepends `job_scripts/slurm_headers/<cluster_name>.slurm` to the job script and submits via `sbatch`. To add a new cluster, just add a new header file. Every submission is logged to `logs/job_scripts/executed_slurm_script_contents.log`.
 
> **SLURM note:** You can also just add a slurm header to the existing bash scripts and execute scripts using sbatch, which is more standard, but this slurm_executor.sh is super helpful while switching between clusters.

---

For a full list of training arguments see `hparams/args.py`.

### Evaluation

#### Linear Probe (SSv2 / ImageNet)

Trains a linear classifier on top of the frozen TDV frame encoder. Frames are pooled via CLS token and optionally concatenated across a short clip. Set `CHECKPOINT_PATH`, `DATASET_DIR`, and the dataset-specific variables at the top of [job_scripts/eval/linear_probe.slurm](job_scripts/eval/linear_probe.slurm) — the script covers both SSv2 and ImageNet.

#### Optical Flow (Sintel)

Uses a Midway decoder ([IterativeLatentMotion](eval/flow/croco/models/latent_motion.py) + DPT head) trained on top of the frozen TDV encoder. The flow eval code lives in [eval/flow/](eval/flow/) and is adapted from the [CroCo v2](https://github.com/naver/croco) stereoflow codebase. See [eval/flow/README.md](eval/flow/README.md) for full setup and dataset download instructions. Set `REPO_ROOT`, `CHECKPOINT`, and the three `dataset_dirs` paths in [job_scripts/eval/flow_eval.slurm](job_scripts/eval/flow_eval.slurm).

#### Stereo Depth (SceneFlow)

Uses the same Midway decoder as optical flow, but in stereo mode (`stereo` subcommand). Set `REPO_ROOT`, `CHECKPOINT`, and `SceneFlow` dataset path in [job_scripts/eval/stereo_depth_eval.slurm](job_scripts/eval/stereo_depth_eval.slurm).

#### Semantic Segmentation (ADE20K / Cityscapes)

Uses a UPerNet decode head on top of the frozen TDV encoder, implemented via the mmsegmentation fork in [repos/mmsegmentation-tdv](repos/mmsegmentation-tdv). The TDV frame encoder is exposed as a drop-in mmseg backbone (`TDVBackbone`) that extracts multi-scale intermediate features. Set `tdv_repo_path` and `data_root` in the config before running (see the repo's README for installation):

```bash
cd repos/mmsegmentation-tdv
python tools/train.py configs/tdv/tdv-base_upernet_160k_ade20k-512x512.py \
  --cfg-options model.backbone.checkpoint_path=/path/to/checkpoint.ckpt \
                model.backbone.tdv_repo_path=/path/to/tdv-clean
```

## General Code Flow

The job script passes all hyperparameters to `train_model.py`. It handles seeding, distributed training setup (DDP), wandb initialization, callback registration (KNN eval, linear probes, DeepSORT tracking), and then calls `trainer.fit()`.

`train_model.py` instantiates `ModelTrainer` from `base_model_trainer.py` — the PyTorch Lightning module responsible for the training loop, validation, optimizer/scheduler setup, gradient logging, and EMA updates. 

### Model Architecture: [model/cv/tdv/tdv.py](model/cv/tdv/tdv.py)

<img src="assets/architecture-whitebg.png" width="75%">

The core model is `TDV`, defined in [`model/cv/tdv/tdv.py`](model/cv/tdv/tdv.py). It implements the temporal difference objective end-to-end and is self-contained — if you only need the architecture and loss, this is the only file you need. 

All hyperparameters live in `hparams/args.py`. Generally you should not need to touch `train_model.py` or `base_model_trainer.py` — new experiments come from adding model variants in `model/cv/` and new datasets in `data/cv/`.


## Repo Structure

```
tdv/
├── claim_tutorial.py                      # interactive collapse-ablation reproduction
├── reports/collapse-ablation/             # methods, results, figures, and limitations
├── train_model.py                          # entry point — parses hparams, sets up DDP, calls trainer
├── base_model_trainer.py                   # PyTorch Lightning module — training loop, optimizer, logging
├── hparams/args.py                         # all hyperparameters
├── model/
│   └── cv/
│       ├── tdv/tdv.py                      # TDV — main model
│       └── dinov2/                         # DINOv2 ViT encoder
├── data/cv/                                # dataloaders (SSv2, Ego4D, FineVideo, Kinetics, ...)
├── eval/
│   ├── flow/                               # optical flow eval (Sintel) — Midway decoder
│   ├── probes/                             # linear probe eval
│   ├── knn/                                # KNN eval
│   └── tracking/                           # DeepSORT tracking eval
├── repos/
│   ├── dino-with-online-knn/               # DINO training with online KNN monitoring
│   ├── ibot-with-online-knn/               # iBOT training with online KNN monitoring
│   └── mmsegmentation-tdv/                 # mmseg fork with TDVBackbone for segmentation eval
├── job_scripts/
│   ├── pretrain_tdv.slurm                  # pretraining launch script
│   ├── slurm_headers/                      # cluster-specific SBATCH headers
│   └── eval/
│       ├── linear_probe.slurm              # linear probe eval (SSv2 / ImageNet)
│       ├── flow_eval.slurm                 # optical flow eval (Sintel / FlyingThings / Chairs)
│       └── stereo_depth_eval.slurm         # stereo depth eval (SceneFlow)
├── requirements.txt
├── CITATION.cff                            # citation metadata for this reproduction artifact
├── THIRD_PARTY_NOTICES.md                  # source and license provenance
└── slurm_executor.sh
```

## Citations

### This artifact

Please cite the independent reproduction artifact using [CITATION.cff](CITATION.cff):

```bibtex
@software{razik_2026_tdv_collapse_ablation,
  author = {Razik-codes},
  title = {TDV Collapse Ablation: A Downscaled Reproduction Artifact},
  year = {2026},
  url = {https://github.com/Razik-codes/tdv-collapse-ablation-reproduction}
}
```

### Original TDV paper

```bibtex
@misc{daithankar2026dontneedstrongassumptions,
      title={You Don't Need Strong Assumptions: Visual Representation Learning via Temporal Differences},
      author={Ninad Daithankar and Alexi Gladstone and Yann LeCun and Heng Ji},
      year={2026},
      eprint={2606.15956},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.15956},
}
```

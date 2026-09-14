#!/usr/bin/env bash
# Downscaled TDV pretraining on a single 4GB-VRAM GPU (NVIDIA RTX 3050 Ti
# Laptop GPU). Originally a pipeline smoke test; now also the reproduction
# harness for the paper's Table-4 collapse ablation. Its collapse-control and
# ablation variants differ only in committed flags below (this file), so every
# node runs the identical `bash job_scripts/pretrain_tdv_local_smoketest.sh`.
#
# This is a scaled-down stand-in for job_scripts/pretrain_tdv.slurm (the
# paper's real multi-GPU SSv2 pretraining launch script), not a hyperparameter
# search: every reduction below exists only to fit 4GB VRAM. See the experiment
# description / project report for the full list of deviations from the paper's
# config and why each was necessary.
#
# Set PYTHON_BIN to the interpreter in an environment provisioned with
# torch/torchvision and `pip install -r requirements.txt --no-deps` (see
# README.md). It defaults to `python`, so an activated virtual environment or
# conda environment works without editing this file.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
	echo "ERROR: Python interpreter $PYTHON_BIN was not found. Activate an environment or set PYTHON_BIN." >&2
	exit 1
fi

# -- respect the 12-thread / 12GB RAM guidance for this machine
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

DATA_DIR="$REPO_ROOT/smoke_data"

# -- generate a tiny synthetic SSv2-format clip set with FAITHFUL low-rank
#    motion (a bright textured disc moving over a static textured background;
#    only ~10% of pixels change between sampled frames). This matches TDV's
#    assumption that Δx is intrinsically low-rank, unlike testsrc2 whose whole
#    frame animates. Runs the real SomethingDataset loader code path unmodified
#    (ffprobe duration + cv2 frame reads + labels/*.json schema); no download.
"$PYTHON_BIN" data/cv/generate_synthetic_motion.py \
	--out_dir "$DATA_DIR" \
	--n_train_per_class 16 \
	--n_val_per_class 4 \
	--skip_if_exists

TDV_ARM="${TDV_ARM:-full}"
case "$TDV_ARM" in
	full)
		export RUN_NAME="tdv-r2-control"
		;;
	no-motion)
		export RUN_NAME="tdv-r2-ablation-no-motion-encoder"
		;;
	no-mse)
		export RUN_NAME="tdv-r2-ablation-no-mse-loss"
		;;
	*)
		echo "ERROR: TDV_ARM must be one of: full, no-motion, no-mse" >&2
		exit 2
		;;
	esac

ARGS=(
	# Run identity
	--modality "CV"
	--model_name "tdv"
	--run_name "${RUN_NAME}"

	# Backbone -- ViT-S/14, smaller of the paper's two studied sizes (ViT-S/ViT-B),
	# from scratch (no weights to download either way; matches the paper's own
	# --load_without_weights default for pretraining)
	--backbone_type "dinov2"
	--vit_backbone_size "small"
	--use_dino_head
	--context_length 2
	--time_between_frames 0.25

	# Frame encoder
	--unfreeze_frame_encoder
	--load_without_weights
	--use_ema_for_frame_encoder
	--ema_momentum 0.990025

	# Difference/motion encoder -- xattn ViT-S variant, depth reduced 12 -> 4 blocks
	--change_encoder_type "dinoViT_xattn_small14"
	--num_transformer_blocks 4
	--difference_encoder_lr_multiplier 3e-2
	--ignore_prefix_tokens_in_condition

	# Loss -- same objective terms + weights as the paper's default config
	--rollout_n_frames 1
	--mse_loss_weight 1.5
	--use_dino_loss
	--dino_loss_weight 0.75
	--use_ibot_loss
	--ibot_loss_weight 0.75
	--recon_loss_type "mse"

	# Teacher / centering / sharpening
	--use_centering
	--use_sharpening
	--dino_teacher_temp 0.1
	--dino_student_temp 0.1
	--ibot_teacher_temp 0.1
	--ibot_student_temp 0.1
	--dino_center_update_momentum 0.9
	--ibot_center_update_momentum 0.9
	--dino_head_prototype_dim 1024

	# Hardware -- single 4GB-VRAM GPU
	--gpus "[0]"
	--set_matmul_precision "medium"
	--float_precision "bf16-mixed"

	# Optimiser -- 600 optimizer steps (max_steps binds; --epochs set high so it
	# is the cap) over the 16 synthetic clips. Long enough for collapse dynamics
	# to show up in the ablations while the full model stays healthy; cosine LR
	# schedule spans the full 600 steps so all arms share the identical schedule.
	--batch_size_per_device 1
	--accumulate_grad_batches 2
	--gradient_clip_val 1.0
	--epochs 600
	--max_steps 600
	--max_scheduling_steps 600
	--peak_learning_rate 1e-4
	--warm_up_steps 0
	--weight_decay 0.01

	# Dataset -- tiny synthetic SSv2-format clip set generated above
	--dataset_name "aggr"
	--aggr_datasets_list "ssv2"
	--aggr_dataset_dirs "${DATA_DIR}"
	--num_workers 2
	--image_dim 224 224

	# Evaluation -- disabled for this pipeline-mechanics smoke test; the real
	# downstream evals (ImageNet KNN, ADE20K/Cityscapes segmentation, Sintel
	# flow, SceneFlow stereo) each need datasets far larger than the ~20GB
	# free on this machine. Validation loss (below) still exercises the
	# held-out eval_step/forward pass end-to-end.
	--run_online_evaluations ""

	# Logging -- wandb off; per-step scalars (incl. representation variance /
	# off-diagonal covariance from --log_var_covar) are printed to stdout with a
	# [METRICS] prefix so the run log is a complete, parseable evidence channel.
	--no_wandb
	--log_var_covar
	--log_baseline_losses
	--print_metrics_to_stdout
	--val_sanity 1
	--log_every_n_steps 1
)

case "$TDV_ARM" in
	full)
		ARGS+=(--use_mse_loss)
		;;
	no-motion)
		ARGS+=(--use_mse_loss --remove_motion_encoder)
		;;
	no-mse)
		# Deliberately omit --use_mse_loss; all remaining flags match the control.
		;;
	esac

"$PYTHON_BIN" train_model.py "${ARGS[@]}"


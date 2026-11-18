#!/bin/bash
# lelab-kit-version: __LELAB_KIT_VERSION__
# LoRA (PEFT) fine-tune of pi0.5 (lerobot/pi05_base, Physical Intelligence's 3B vision-language-action model): low-rank
# adapters on the action expert, in bfloat16 with gradient checkpointing, so less GPU memory and small checkpoints.
# Runs on a dataset LeLab sent to this server. Settings come from .env in the project folder (PI05_*, PI05_LORA_* and
# the shared ones; see .env.example). Submit from anywhere:
#
#   sbatch {project}/slurm/train_pi05_lora.sbatch
#   DATASET=<folder in DATASET_DIR> sbatch {project}/slurm/train_pi05_lora.sbatch     # one job on another dataset, .env unchanged
#
# The run trains in $OUTPUT_DIR/pi05_lora_<dataset>[_<RUN_TAG>]; submitting it again resumes it. When it finishes,
# the policy is published to $POLICY_DIR/pi05_lora_<dataset>, ready for slurm/serve_policy.sbatch.
# pi05_base's tokenizer (google/paligemma-3b-pt-224) is gated: accept its license on huggingface.co and set HF_TOKEN.
#
# The resources below match the student account on the B200 cluster (the so-101 arm project's slurm/ scripts):
# gpu-small is a 23 GB GPU slice; 6 CPUs and 32 GB are the student limits. Change them here for another cluster.
# Keep #SBATCH lines free of trailing comments: the cluster's Slurm Guard reads them as part of the value.
#
#SBATCH --job-name=lelab-pi05-lora
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# LeLab writes the project folder in here when it sends these files; LELAB_PROJECT_DIR overrides it.
PROJECT_DIR=${LELAB_PROJECT_DIR:-'__LELAB_PROJECT_DIR__'}
case $PROJECT_DIR in __LELAB*) PROJECT_DIR=${SLURM_SUBMIT_DIR:-$PWD} ;; esac
POLICY_KIND=pi05
FINETUNE=lora
# shellcheck source=lelab_train.sh
source "$PROJECT_DIR/slurm/lelab_train.sh"

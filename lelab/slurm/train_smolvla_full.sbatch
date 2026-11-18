#!/bin/bash
# lelab-kit-version: __LELAB_KIT_VERSION__
# Full fine-tune of SmolVLA (lerobot/smolvla_base): every weight trains, vision encoder included, unless .env freezes it.
# Runs on a dataset LeLab sent to this server. Settings come from .env in the project folder (SMOLVLA_* and the shared
# ones; see .env.example). Submit from anywhere:
#
#   sbatch {project}/slurm/train_smolvla_full.sbatch
#   DATASET=<folder in DATASET_DIR> sbatch {project}/slurm/train_smolvla_full.sbatch     # one job on another dataset, .env unchanged
#
# The run trains in $OUTPUT_DIR/smolvla_full_<dataset>[_<RUN_TAG>]; submitting it again resumes it. When it finishes,
# the policy is copied to $POLICY_DIR/smolvla_full_<dataset>, ready for slurm/serve_policy.sbatch.
#
# The resources below match the student account on the B200 cluster (the so-101 arm project's slurm/ scripts):
# gpu-small is a 23 GB GPU slice; 6 CPUs and 32 GB are the student limits. Change them here for another cluster.
# Keep #SBATCH lines free of trailing comments: the cluster's Slurm Guard reads them as part of the value.
#
#SBATCH --job-name=lelab-smolvla-full
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
POLICY_KIND=smolvla
FINETUNE=full
# shellcheck source=lelab_train.sh
source "$PROJECT_DIR/slurm/lelab_train.sh"

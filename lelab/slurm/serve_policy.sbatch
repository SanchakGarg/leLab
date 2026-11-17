#!/bin/bash
# Serve a trained policy on a GPU node, so LeLab on your computer can run it on the arm (Run model on server).
# Runs lelab_policy_server.py (LeLab sends it next to these scripts) with the policy named by POLICY in .env:
#
#   sbatch {project}/slurm/serve_policy.sbatch
#   POLICY=policies/act_full_pick_cube sbatch {project}/slurm/serve_policy.sbatch   # another one, .env unchanged
#   tail -f lelab-serve-<JOBID>.out            # prints "serving on <ip>:<port>" once it is ready
#
# POLICY is a pretrained_model folder (a run in POLICY_DIR, or any .../checkpoints/<step>/pretrained_model; LoRA
# runs work too) or a Hub repo id. Empty: the newest run in POLICY_DIR. Also read from .env: SERVE_PORT (5555; the
# next free port is used if it is taken), SERVE_DEVICE (cuda), SERVE_N_ACTION_STEPS (play only the first N actions
# of each chunk; default the policy's own). In LeLab, enter the node IP and port the log prints.
# The job serves until its time limit or scancel <JOBID>.
#
# Resources as in the so-101 arm project's serve script. Keep #SBATCH lines free of trailing comments: the cluster's
# Slurm Guard reads them as part of the value.
#
#SBATCH --job-name=lelab-serve
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

PROJECT_DIR=${LELAB_PROJECT_DIR:-'__LELAB_PROJECT_DIR__'}
case $PROJECT_DIR in __LELAB*) PROJECT_DIR=${SLURM_SUBMIT_DIR:-$PWD} ;; esac
# shellcheck source=lelab_common.sh
source "$PROJECT_DIR/slurm/lelab_common.sh"

SERVER=$PROJECT_DIR/lelab_policy_server.py
[ -f "$SERVER" ] || die "no $SERVER. Send the Slurm files again from LeLab (Remote server -> Send Slurm files)."

if [ -z "${POLICY:-}" ]; then                 # the most recently finished training run
    POLICY=$(ls -dt "$POLICY_DIR"/*/ 2>/dev/null | head -1 || true)
    POLICY=${POLICY%/}
    [ -n "$POLICY" ] || die "no trained policy in $POLICY_DIR yet. Train one (slurm/train_*.sbatch) or set POLICY" \
        "in $ENV_FILE."
    echo "=== POLICY is empty: serving the newest run, $POLICY ==="
elif [ -L "$(lelab_path "$POLICY")" ] && [ ! -e "$(lelab_path "$POLICY")" ]; then
    die "$(lelab_path "$POLICY") links to $(readlink "$(lelab_path "$POLICY")"), which no longer exists (its" \
        "training run was deleted). Train it again, or train with POLICY_COPY=1 to keep a copy."
elif [ -e "$(lelab_path "$POLICY")" ]; then
    POLICY=$(lelab_path "$POLICY")
else
    case $POLICY in
        */*/* | /* | ./* | ../*) die "no policy at $(lelab_path "$POLICY")" ;;
        */*) echo "=== $POLICY is not a folder here: loading it from the Hugging Face Hub ===" ;;
        *) die "no policy at $(lelab_path "$POLICY")" ;;
    esac
fi
if [ -d "$POLICY" ] && [ ! -f "$POLICY/config.json" ] && [ ! -f "$POLICY/adapter_config.json" ]; then
    if [ -f "$POLICY/checkpoints/last/pretrained_model/config.json" ] \
        || [ -f "$POLICY/checkpoints/last/pretrained_model/adapter_config.json" ]; then
        POLICY=$POLICY/checkpoints/last/pretrained_model   # a training run folder: its last checkpoint
    else
        die "$POLICY holds no policy (config.json or adapter_config.json)"
    fi
fi

ARGS=(--policy "$POLICY" --host 0.0.0.0 --port "${SERVE_PORT:-5555}" --device "${SERVE_DEVICE:-cuda}")
if [ -n "${SERVE_N_ACTION_STEPS:-}" ]; then ARGS+=(--n-action-steps "$SERVE_N_ACTION_STEPS"); fi
echo "=== serving $POLICY; stop it with: scancel ${SLURM_JOB_ID:-<job id>} ==="
exec "$PY" "$SERVER" "${ARGS[@]}"

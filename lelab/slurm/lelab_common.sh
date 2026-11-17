# shellcheck shell=bash
# Shared by LeLab's Slurm scripts (slurm/*.sbatch): reads the project's .env, resolves the folders it names and
# finds the Python environment. Sourced at the top level, after the script has set PROJECT_DIR.
#
# Settings come from <project>/.env (copy .env.example to .env and edit it). A variable given on the command line
# wins over .env, so one job can differ without editing it:
#   DATASET=pick_cube_2 sbatch slurm/train_act.sbatch

set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

ENV_FILE=${LELAB_ENV_FILE:-$PROJECT_DIR/.env}
if [ -f "$ENV_FILE" ]; then
    _lelab_preset=$(export -p)               # what the job was submitted with...
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    eval "$_lelab_preset" 2>/dev/null || true # ...wins over .env
    unset _lelab_preset
    echo "=== settings from $ENV_FILE ==="
    grep -v '^[[:space:]]*\(#\|$\)' "$ENV_FILE" || true
else
    echo "=== no $ENV_FILE: using the defaults (cp $PROJECT_DIR/.env.example $ENV_FILE to change them) ==="
fi

# A folder from .env: absolute, ~/..., or relative to the project folder.
lelab_path() {
    case $1 in
        /*) printf '%s\n' "$1" ;;
        "~") printf '%s\n' "$HOME" ;;
        "~/"*) printf '%s\n' "$HOME/${1#"~/"}" ;;
        *) printf '%s\n' "$PROJECT_DIR/$1" ;;
    esac
}

LELAB_ENV_DIR=$(lelab_path "${LELAB_ENV_DIR:-venv}")
DATASET_DIR=$(lelab_path "${DATASET_DIR:-datasets}")
OUTPUT_DIR=$(lelab_path "${OUTPUT_DIR:-outputs}")
POLICY_DIR=$(lelab_path "${POLICY_DIR:-policies}")
HF_HOME=$(lelab_path "${HF_HOME:-hf_cache}")
export HF_HOME HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# Optional: modules to load and a command to run first (e.g. activating conda), from .env.
if [ -n "${LELAB_MODULES:-}" ]; then
    # shellcheck disable=SC2086
    module load $LELAB_MODULES
fi
if [ -n "${LELAB_SETUP:-}" ]; then
    eval "$LELAB_SETUP"
fi

PY=$LELAB_ENV_DIR/bin/python
[ -x "$PY" ] || die "no Python at $PY. Set LELAB_ENV_DIR in $ENV_FILE to the venv or conda env with LeRobot" \
    "(the folder holding bin/python), or create it there (see .env.example)."

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname), partition ${SLURM_JOB_PARTITION:-?}, project $PROJECT_DIR ==="
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L || true; fi

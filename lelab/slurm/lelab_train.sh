# shellcheck shell=bash
# Trains one LeRobot policy on a dataset LeLab sent to this server. Sourced by slurm/train_*.sbatch, which set
#   POLICY_KIND = act | smolvla
#   FINETUNE    = full | lora
# Every setting comes from .env (see .env.example); a variable on the sbatch command line wins over it.
#
# Each run trains in $OUTPUT_DIR/<kind>_<finetune>_<dataset>[_<RUN_TAG>]. Submitting the same job again resumes it
# from its last checkpoint (after a time limit or scancel). When it finishes, the trained policy is copied to
# $POLICY_DIR/<run>, which slurm/serve_policy.sbatch serves.

# shellcheck source=lelab_common.sh
source "$PROJECT_DIR/slurm/lelab_common.sh"

case $POLICY_KIND in act | smolvla) ;; *) die "POLICY_KIND must be act or smolvla, got '$POLICY_KIND'" ;; esac
case $FINETUNE in full | lora) ;; *) die "FINETUNE must be full or lora, got '$FINETUNE'" ;; esac

# ---- dataset ----------------------------------------------------------------------------------------------------------
[ -n "${DATASET:-}" ] || die "set DATASET in $ENV_FILE to a dataset folder in $DATASET_DIR (or an absolute path)," \
    "or submit with DATASET=<name> sbatch ..."
case $DATASET in /*) DATA=$DATASET ;; *) DATA=$DATASET_DIR/$DATASET ;; esac
DATA=${DATA%/}
[ -f "$DATA/meta/info.json" ] || die "no LeRobot dataset at $DATA. Send it from LeLab (Send to server) into" \
    "$DATASET_DIR, or fix DATASET / DATASET_DIR."
DATA_NAME=$(basename "$DATA")
REPO_ID=${DATASET_REPO_ID:-local/$DATA_NAME}

# ---- shared settings --------------------------------------------------------------------------------------------------
DEVICE=${DEVICE:-cuda}
SAVE_FREQ=${SAVE_FREQ:-10000}
LOG_FREQ=${LOG_FREQ:-200}
NUM_WORKERS=${NUM_WORKERS:-4}
AUGMENT=${AUGMENT:-0}
RUN=${POLICY_KIND}_${FINETUNE}_${DATA_NAME}${RUN_TAG:+_$RUN_TAG}
OUT=$OUTPUT_DIR/$RUN
LAST=$OUT/checkpoints/last/pretrained_model/train_config.json
TRAIN=${LEROBOT_TRAIN:-$LELAB_ENV_DIR/bin/lerobot-train}
[ -x "$TRAIN" ] || die "no lerobot-train in $LELAB_ENV_DIR/bin. Install LeRobot there (see .env.example)."

"$PY" -c "import torch, lerobot; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || die "torch or lerobot doesn't import with $PY"
if [ "$FINETUNE" = lora ] && ! "$PY" -c "import peft" 2>/dev/null; then
    die "LoRA needs the peft package in $LELAB_ENV_DIR: $PY -m pip install 'lerobot[peft]' (or uv pip install" \
        "--python $PY 'lerobot[peft]')"
fi

ARGS=()
add() { ARGS+=("$@"); }

# ---- per policy -------------------------------------------------------------------------------------------------------
if [ "$POLICY_KIND" = act ]; then
    STEPS=${ACT_STEPS:-100000}
    BATCH=${ACT_BATCH:-32}
    ACT_INIT=${ACT_INIT:-}
    if [ "$FINETUNE" = full ]; then
        LR=${ACT_LR:-1e-5}
        if [ -n "$ACT_INIT" ]; then           # continue training an ACT you already have (its chunk size stays)
            add --policy.path="$ACT_INIT"
            DESC="ACT, all weights, starting from $ACT_INIT"
        else
            add --policy.type=act --policy.chunk_size="${ACT_CHUNK:-100}"
            DESC="ACT from scratch, chunk ${ACT_CHUNK:-100}"
        fi
        add --policy.optimizer_lr="$LR" --policy.optimizer_lr_backbone="$LR"
        if [ -n "${ACT_N_ACTION_STEPS:-}" ]; then add --policy.n_action_steps="$ACT_N_ACTION_STEPS"; fi
    else
        [ -n "$ACT_INIT" ] || die "LoRA adapts a trained ACT: set ACT_INIT to its pretrained_model folder (e.g. a" \
            "run in $POLICY_DIR) or a Hub repo id. To train ACT from scratch use slurm/train_act.sbatch."
        LR=${ACT_LORA_LR:-1e-4}
        # ACT's transformer feed-forward layers (attention runs through nn.MultiheadAttention, which LoRA can't wrap);
        # the action head trains fully.
        TARGETS=${ACT_LORA_TARGETS:-'model\.(encoder|decoder)\.layers\.\d+\.(linear1|linear2)'}
        add --policy.path="$ACT_INIT" --policy.optimizer_lr="$LR" --policy.optimizer_lr_backbone="$LR"
        add --peft.method_type=LORA --peft.r="${ACT_LORA_R:-16}" --peft.lora_alpha="${ACT_LORA_ALPHA:-32}"
        add --peft.target_modules="$TARGETS" --peft.full_training_modules="${ACT_LORA_FULL_MODULES:-[action_head]}"
        DESC="ACT LoRA r${ACT_LORA_R:-16} on $ACT_INIT"
    fi
else
    STEPS=${SMOLVLA_STEPS:-30000}
    BATCH=${SMOLVLA_BATCH:-32}
    BASE=${SMOLVLA_BASE:-lerobot/smolvla_base}
    add --policy.path="$BASE" --policy.scheduler_decay_steps="$STEPS"
    if [ "$FINETUNE" = full ]; then
        FREEZE=${SMOLVLA_FREEZE_VISION:-false}
        EXPERT_ONLY=${SMOLVLA_EXPERT_ONLY:-false}
        add --policy.optimizer_lr="${SMOLVLA_LR:-1e-4}" --policy.scheduler_decay_lr="${SMOLVLA_DECAY_LR:-2.5e-6}"
        add --policy.freeze_vision_encoder="$FREEZE" --policy.train_expert_only="$EXPERT_ONLY"
        DESC="SmolVLA full fine-tune from $BASE (freeze vision $FREEZE, expert only $EXPERT_ONLY)"
    else
        add --policy.optimizer_lr="${SMOLVLA_LORA_LR:-1e-3}" \
            --policy.scheduler_decay_lr="${SMOLVLA_LORA_DECAY_LR:-1e-4}"
        add --peft.method_type=LORA --peft.r="${SMOLVLA_LORA_R:-64}" --peft.lora_alpha="${SMOLVLA_LORA_ALPHA:-64}"
        case ${SMOLVLA_LORA_TARGET:-expert} in
            expert) ;;                         # LeRobot's default: the action expert's q/v + state/action projections
            vlm) add '--peft.target_modules=(model\.vlm_with_expert\.(lm_expert|vlm\.model\.text_model|vlm\.model\.vision_model\.encoder)\.layers\.\d+\.self_attn\.(q|v)_proj|model\.vlm_with_expert\.vlm\.model\.connector\.modality_projection\.proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))' ;;
            *) die "SMOLVLA_LORA_TARGET must be expert or vlm, got '$SMOLVLA_LORA_TARGET'" ;;
        esac
        DESC="SmolVLA LoRA r${SMOLVLA_LORA_R:-64} on ${SMOLVLA_LORA_TARGET:-expert} from $BASE"
    fi
    # smolvla_base was trained on cameras named camera1, camera2, camera3: map the dataset's cameras onto them.
    if [ "${SMOLVLA_RENAME_MAP:-}" = none ]; then
        RENAME=
    elif [ -n "${SMOLVLA_RENAME_MAP:-}" ]; then
        RENAME=$SMOLVLA_RENAME_MAP
    else
        RENAME=$("$PY" - "$DATA/meta/info.json" "${SMOLVLA_CAMERAS:-}" <<'PYEOF'
import json, sys
features = json.load(open(sys.argv[1]))["features"]
cams = [k for k in features if k.startswith("observation.images.")]
order = [c.strip() for c in sys.argv[2].split(",") if c.strip()]
if order:
    want = [c if c.startswith("observation.images.") else "observation.images." + c for c in order]
    missing = [w for w in want if w not in cams]
    if missing:
        sys.exit(f"SMOLVLA_CAMERAS names cameras the dataset doesn't have: {missing}; it has {cams}")
    cams = want
if len(cams) > 3:
    print(f"WARNING: smolvla_base takes 3 cameras; using {cams[:3]} (pick them with SMOLVLA_CAMERAS)", file=sys.stderr)
print(json.dumps({c: f"observation.images.camera{i + 1}" for i, c in enumerate(cams[:3])}))
PYEOF
        )
    fi
    if [ -n "$RENAME" ]; then
        add --rename_map="$RENAME"
        echo "=== cameras: $RENAME ==="
    fi
fi

if [ "$AUGMENT" = 1 ]; then add --dataset.image_transforms.enable=true; fi
if [ -n "${SEED:-}" ]; then add --seed="$SEED"; fi

# ---- train (or resume) ------------------------------------------------------------------------------------------------
# ---- Trackio: loss, learning rate and speed per run, in TRACKIO_DIR (or a Hugging Face Space) --------------------------
TRACK_ARGS=()
if [ "${TRACKIO:-1}" = 1 ]; then
    if "$PY" -c "import trackio" 2>/dev/null; then
        TRACKIO_DIR=$(lelab_path "${TRACKIO_DIR:-trackio}")
        export TRACKIO_DIR
        TRACK_ARGS=(--trackio.enable=true --trackio.project="${TRACKIO_PROJECT:-lelab}")
        if [ -n "${TRACKIO_SPACE_ID:-}" ]; then
            TRACK_ARGS+=(--trackio.space_id="$TRACKIO_SPACE_ID" --trackio.private=true)
            echo "=== Trackio: project ${TRACKIO_PROJECT:-lelab}, synced to the Space $TRACKIO_SPACE_ID ==="
        else
            echo "=== Trackio: project ${TRACKIO_PROJECT:-lelab}, runs in $TRACKIO_DIR" \
                "(view: TRACKIO_DIR=$TRACKIO_DIR $LELAB_ENV_DIR/bin/trackio show --project ${TRACKIO_PROJECT:-lelab}) ==="
        fi
    else
        echo "WARNING: TRACKIO=1, but trackio isn't installed in $LELAB_ENV_DIR, so this run isn't tracked." \
            "Install it with: $PY -m pip install 'lerobot[training]' (or set TRACKIO=0)." >&2
    fi
else
    TRACK_ARGS=(--trackio.enable=false)
fi

mkdir -p "$OUTPUT_DIR" "$POLICY_DIR"
if [ -f "$LAST" ]; then
    echo "=== resuming $OUT from $(readlink "$OUT/checkpoints/last" || echo last), to step $STEPS ==="
    CMD=("$TRAIN" --config_path="$LAST" --resume=true --steps="$STEPS" --save_freq="$SAVE_FREQ")
else
    if [ -d "$OUT" ]; then                   # cancelled before its first checkpoint: nothing to resume
        echo "=== $OUT has no checkpoint yet; moving it to $OUT.incomplete-${SLURM_JOB_ID:-$$} and starting fresh ==="
        mv "$OUT" "$OUT.incomplete-${SLURM_JOB_ID:-$$}"
    fi
    echo "=== new run $OUT: $DESC; $STEPS steps, batch $BATCH, dataset $DATA ==="
    CMD=("$TRAIN" "${ARGS[@]}"
        --dataset.repo_id="$REPO_ID" --dataset.root="$DATA"
        --policy.device="$DEVICE" --policy.push_to_hub=false
        --output_dir="$OUT" --job_name="$RUN"
        --batch_size="$BATCH" --steps="$STEPS"
        --save_freq="$SAVE_FREQ" --log_freq="$LOG_FREQ" --num_workers="$NUM_WORKERS")
fi
CMD+=(${TRACK_ARGS[@]+"${TRACK_ARGS[@]}"})
if [ -n "${EXTRA_TRAIN_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    CMD+=($EXTRA_TRAIN_ARGS)
fi
"${CMD[@]}"

# ---- publish the trained policy ---------------------------------------------------------------------------------------
SRC=$OUT/checkpoints/last/pretrained_model
[ -d "$SRC" ] || die "training ended without $SRC"
DEST=$POLICY_DIR/$RUN
rm -rf "$DEST.tmp"
cp -RL "$SRC" "$DEST.tmp"
rm -rf "$DEST"
mv "$DEST.tmp" "$DEST"
echo "=== done: trained policy in $DEST ==="
echo "Serve it: POLICY=$DEST sbatch $PROJECT_DIR/slurm/serve_policy.sbatch   (or set POLICY in .env)"

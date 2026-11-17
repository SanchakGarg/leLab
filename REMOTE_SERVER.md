# Your own GPU server: record here, train there, run the model on this arm

LeLab can work with a GPU machine you log into over SSH: a workstation, a cloud VM or a Slurm cluster.

1. **Save the server** (user name and IP) and connect.
2. **Record** a dataset as usual. LeLab copies it to a folder on the server over SFTP.
3. **Train** on the server yourself, with `lerobot-train`.
4. **Serve** the trained model there with `lelab_policy_server.py`, then enter its IP and port in LeLab. LeLab connects through your SSH login and runs the model on the real follower arm.

The model runs on the server. The control loop runs on this computer, next to the arm.

## 1. Save and connect the server

Click the server chip in the top bar (**Add server**). Fill in:

| field | example | meaning |
|---|---|---|
| Name | `cluster` | any name for this server |
| IP address or host | `10.2.100.14` | where you `ssh` to (the login node on a cluster) |
| Username, SSH port | `student`, `22` | as in `ssh -p 22 student@10.2.100.14` |
| Log in with | Password | **Password**, a **Key file** (`~/.ssh/id_ed25519`), or **ssh-agent / default keys** |
| Dataset folder on the server | `projects/lelab/data` | where datasets go; relative paths are under your home |
| Work folder on the server | `projects/lelab` | the project folder: `.env`, the Slurm jobs and the policy server script go here |

Save, then **Connect**.

- **Your password is never saved.** LeLab keeps it in memory while it runs, so a dropped connection reconnects without asking. After a LeLab restart (`lelab --dev` restarts on every code change) you enter it again; key and ssh-agent logins reconnect by themselves.
- **With several servers,** LeLab uses the one you last picked, connected to or sent to, else the first saved. The send dialog lists them all with their connection state.
- **The first time you connect, LeLab shows the server's host key fingerprint.** Trust it only if it is your server's. Ask the administrator, or compare it with `ssh-keygen -lf /etc/ssh/ssh_host_*_key.pub` on the server. Keys you already accepted with your own `ssh` (in `~/.ssh/known_hosts`) are trusted already.
- **Saved profiles** are in `~/.cache/huggingface/lerobot/remote_servers/`. Trusted keys are in `~/.cache/huggingface/lerobot/remote_known_hosts`.

## 2. Send a dataset to the server

Choose one of these:
- **Automatically:** tick **Send to *server* when done** in the recording dialog. The dataset goes to the server as soon as the last episode is saved.
- **By hand:** use **Send to server** on the page after recording, or in the dataset browser for any local dataset.

The dataset lands at `<dataset folder>/<dataset name>` with the usual LeRobot layout (`meta/`, `data/`, `videos/`).

It goes over SFTP, not scp, because some cluster shells refuse scp. You can cancel and send again: files already on the server are skipped, so an interrupted upload resumes where it stopped.

The dialog shows the server's real connection state: it checks again when it opens and every few seconds while it is open. If the session is gone (LeLab restarted, or the connection dropped), it shows the password field again instead of **Connected**.

**Also send the Slurm training files** (ticked by default, remembered) copies the training kit to the work folder in the same go (see step 3). You can also send it alone with **Send Slurm training files to the work folder** in the server dialog.

When it's done, LeLab shows the `sbatch` commands that train on the dataset (or a `lerobot-train` command when the Slurm files aren't on the server).

## 3. Train on the server

### With the Slurm jobs LeLab sends

LeLab puts this in the work folder (the scripts live in [lelab/slurm/](lelab/slurm/)):

```
<work folder>/
  .env.example               every setting, with comments; replaced on each send
  .env                       your settings: created from .env.example only if missing, never overwritten
  lelab_policy_server.py
  slurm/train_act.sbatch           ACT, all weights: from scratch, or from ACT_INIT to keep training one
  slurm/train_act_lora.sbatch      LoRA on a trained ACT (ACT_INIT)
  slurm/train_smolvla_full.sbatch  SmolVLA full fine-tune from lerobot/smolvla_base
  slurm/train_smolvla_lora.sbatch  SmolVLA LoRA (PEFT) fine-tune
  slurm/serve_policy.sbatch        serves POLICY with lelab_policy_server.py
  slurm/lelab_common.sh, lelab_train.sh   shared by the jobs
```

Set these in `.env` once:

| setting | default | meaning |
|---|---|---|
| `LELAB_ENV_DIR` | `venv` | venv or conda env with LeRobot (the folder holding `bin/python`). For LoRA it needs `lerobot[peft]`. |
| `DATASET_DIR` | the dataset folder you sent to | where the datasets are |
| `DATASET` | | the dataset to train on: a folder in `DATASET_DIR`, or an absolute path |
| `OUTPUT_DIR` | `outputs` | training runs, one folder per run: `<policy>_<full\|lora>_<dataset>[_<RUN_TAG>]` |
| `POLICY_DIR` | `policies` | each finished run's last checkpoint is copied to `POLICY_DIR/<run>` |
| `POLICY` | newest in `POLICY_DIR` | what `serve_policy.sbatch` serves |

Relative folders are inside the work folder. The training settings (steps, batch size, learning rates, LoRA rank and targets, which camera is SmolVLA's `camera1`, ...) are in `.env.example`, grouped per job. Then, from a login shell:

```bash
sbatch ~/projects/lelab/slurm/train_smolvla_lora.sbatch
DATASET=pick_cube_2 sbatch ~/projects/lelab/slurm/train_act.sbatch   # a command-line value wins over .env
```

Submitting the same job again resumes the run from its last checkpoint (after a time limit or `scancel`). The `#SBATCH` lines are set for the student account of the B200 cluster (partition `gpu-small`, 6 CPUs, 32 GB, 1 day), as in the so-101 arm project; edit them in the scripts for another cluster.

### By hand

Train however you like. `--dataset.root` is the folder the upload printed:

```bash
lerobot-train \
  --dataset.repo_id=local/pick_cube_20261117_120000 \
  --dataset.root=$HOME/projects/lelab/data/pick_cube_20261117_120000 \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/act_pick_cube
```

LeLab records the follower in degrees (gripper 0–100). A model trained on these datasets predicts in the same units, which is what LeLab sends to the arm.

## 4. Serve the model and run it on the arm

**Copy the script.** The Slurm training files include it. Or, in the server dialog, click **Copy the policy server script to the work folder**, or install LeLab on the server and use the `lelab-policy-server` command instead.

The script needs only the Python you train with: `lerobot`, `torch`, `numpy`, `Pillow`, plus `peft` for LoRA checkpoints.

**Start it on a GPU machine.** With the Slurm files, set `POLICY` in `.env` (a folder in `policies/`, any `pretrained_model` folder, or a Hub repo id; empty serves the newest trained policy) and run:

```bash
sbatch ~/projects/lelab/slurm/serve_policy.sbatch
POLICY=policies/act_full_pick_cube sbatch ~/projects/lelab/slurm/serve_policy.sbatch   # or pick one per job
```

`SERVE_PORT`, `SERVE_DEVICE` and `SERVE_N_ACTION_STEPS` in `.env` set the rest. Without the Slurm files, a job like this one does the same (adjust the partition, account and paths):

```bash
#!/bin/bash
#SBATCH --job-name=lelab-serve
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=lelab-serve-%j.out
cd ~/projects/lelab
venv/bin/python lelab_policy_server.py \
  --policy outputs/act_pick_cube/checkpoints/last/pretrained_model --port 5555
```

```bash
sbatch serve.sbatch
tail -f lelab-serve-<JOBID>.out
```

The log shows the model's cameras and chunk size, three warm-up inferences, and then a line like this:

```
=== serving on 10.2.0.7:5555 ===
```

- **10.2.0.7** is the GPU node's IP.
- **5555** is the port. If 5555 is taken on that node, the server uses the next free port and the log says so.

**Connect from LeLab.** In **Jobs**, click **Run model on server**:

1. **Connect: Through the server's SSH login (tunnel).**
   - **GPU node IP:** the IP from the log. Use `127.0.0.1` if the server runs on the machine you log into.
   - **Port:** the port from the log.

   This is what `ssh -N -L 5555:10.2.0.7:5555 student@10.2.100.14` does, without opening a local port.

   If your computer is on the same network as the GPU node, **Directly** works too.
2. **Connect to the model.** LeLab shows the model and the cameras it was trained with, at their resolutions.
3. **Bind a camera to each one.** A robot camera with the same name binds itself. Use the same rotation as in the training data.
4. **Set the task** (for language-conditioned models) and the duration.
5. **Optionally tick Record a video of the run** (see below), then **Start**.

**While it runs**, the Inference page shows the round trip (network plus inference) and how long the arm waited for the server.

How it keeps the arm moving:
- LeLab asks for the next chunk while 10 actions are still queued (**Prefetch**). The arm keeps moving as long as one round trip is shorter than those actions take to play.
- It drops the actions already played while a request was out, so each new chunk lines up with the present.
- **Actions per chunk** replans more often.

## Record a video of the run

Tick **Record a video of the run** before you start. LeLab saves an mp4 of what the model's cameras saw, with one camera full frame and another inset in the bottom-right corner.

- **Two cameras, one of them the wrist camera:** the other camera is full frame and the wrist camera goes in the corner. A camera counts as the wrist camera when its name, or the name of the robot camera bound to it, contains `wrist`, `hand` or `gripper`.
- **More than two cameras, or two without a wrist camera:** pick the **Full view** camera and the one **In the corner** (or None).
- **One camera:** it fills the frame.

The video covers the run from the first action to the last, at the control rate. It is written on this computer while the arm moves, without slowing the arm down: if encoding falls behind, frames are dropped.

When the run ends, the Inference page plays the video and offers **Download**. The files are in `~/.cache/huggingface/lerobot/run_videos/`, named after the date, time and checkpoint folder (`20261117_120000_act_pick_cube.mp4`).

## Safety

- **Max move per step** (default 20°) caps how far any joint moves in one step.
- **The run stops on any fault:**
  - an error from the model
  - a dropped connection or job
  - a chunk that doesn't arrive within 5 s (60 s for the first, which can include warm-up)
  - a bad chunk (wrong shape, NaN)
  - a camera that stops delivering frames
- **After a stop, the arm returns to its start pose.** On a fault, Stop, or the end of the duration, the arm moves back to where it started over 3 s, then its motors are released (torque off), as in a local LeLab run. If a motor command itself failed, the arm is released where it is, so support it.
- **The server takes one client at a time.** A second connection is told it is busy.
- **Brief USB glitches don't stop the run.** A failed read of the follower reuses its last positions, and a failed write is skipped (the motors hold their last goal). The run stops only when the arm hasn't answered for 10 s, and then the arm is released where it is, so support it. This also applies to teleoperation, recording and replay, for both arms.

## Troubleshooting

| message | what to do |
|---|---|
| Login … failed | Check the user name and password. A key the server doesn't know is refused too. |
| *host* is not a known host yet | First connection: check the fingerprint, then **Trust and connect**. |
| Nothing answers at *IP:port* as seen from *server* | The job isn't running yet, or the IP or port differs from the log. |
| … does not allow SSH tunnels | The login node refuses forwarding. Use **Directly** from the same network, or ask the administrator. |
| Connected, but the policy server said nothing within 10 s | It is still loading the model (wait for `serving on`), or another client is connected. |
| … refused SFTP | The server has no SFTP. Ask the administrator how to copy files to it. |
| The connection to *server* is not open (LeLab may have restarted) | LeLab forgot the session. Enter the password and **Connect**, then **Send** again. |
| `ERROR: no Python at .../bin/python` in a job log | Set `LELAB_ENV_DIR` in `.env` to your LeRobot venv or conda env. |
| `ERROR: set DATASET` in a job log | Set `DATASET` in `.env`, or submit with `DATASET=<name> sbatch ...`. |

## The protocol

`lelab_policy_server.py` speaks the TCP protocol below. Any client that implements it can drive a model served this way.

Every message has the same framing:
- two big-endian uint32 values: the header length and the payload length
- a JSON header
- the payload bytes

The messages are:

| direction | message | payload | reply |
|---|---|---|---|
| server → client, on connect | `hello` with `name`, `cameras`, `image_shapes`, `state_dim`, `action_dim`, `chunk_size`, `n_action_steps`, `fps` | — | — |
| client → server | `obs` with `id`, `task`, `state`, `images: {camera: [offset, length]}` | JPEG frames | `chunk` with `id` and `shape: [T, 6]` (float32 actions as the payload), or `error` with `message` |
| client → server | `reset` (clears the policy's state) | — | none |

The server source and its docstring are in [`lelab/policy_server.py`](lelab/policy_server.py).

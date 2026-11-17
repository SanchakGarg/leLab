# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

LeLab is a FastAPI + React web interface wrapping the [LeRobot](https://github.com/huggingface/lerobot) framework for the SO-101 leader/follower arm. It exposes teleoperation, dataset recording, calibration, replay, and training as HTTP/WebSocket endpoints, replacing LeRobot's CLI + keyboard-driven flows.

The frontend (React + Vite) lives in [`frontend/`](frontend/). The bundle in `frontend/dist/` is not committed: [setup.py](setup.py) builds it while building the wheel, with Node from the `nodejs-wheel-binaries` build requirement, and ships it as package data (`frontend.__init__.py` makes setuptools treat it as a package); [`lelab/server.py`](lelab/server.py) mounts it as `StaticFiles` at `/` so a single `lelab` process serves both API and UI on `:8000`. The `frontend/` directory is also force-pushed to the [LeLab HF Space](https://huggingface.co/spaces/lerobot/LeLab) by [`.github/workflows/sync_space.yml`](.github/workflows/sync_space.yml) — that's a separate runtime; the Space builds the [Dockerfile](frontend/Dockerfile) (which runs `npm run build` itself).

## Common commands

Install (editable, requires Python ≥3.10):

```bash
pip install -e .
```

`lerobot` is pinned to a ref on `huggingface/lerobot` (currently `v0.7.0`, see [pyproject.toml](pyproject.toml)), installed from git rather than PyPI because the published wheels don't expose the `lerobot-rollout` script that [lelab/rollout.py](lelab/rollout.py) shells out to. Bump the ref when you want a newer upstream release; expect import-path drift (e.g. `lerobot.configs.dataset`, `lerobot.utils.feature_utils`, `lerobot.common.control_utils`) and CLI flag renames (v0.7.0 replaced `--wandb.*` with `--trackio.*`). [tests/test_lerobot_cli.py](tests/test_lerobot_cli.py) runs the training and rollout argv LeLab builds through the installed LeRobot's parsers, so run the tests right after a bump. Since v0.7.0 the arms' motors go through LeRobot's rustypot-backed `SerialMotorsBus`: use its public methods (`set_operating_mode`, `set_half_turn_homings`, `disconnect`), not Feetech SDK internals.

Run servers (entry point defined in [pyproject.toml](pyproject.toml)):

```bash
lelab          # uvicorn on :8000, serves built frontend at /, opens browser
lelab --dev    # spawns Vite dev (:8080) + uvicorn --reload (:8000), opens browser to :8080
```

Editable installs skip the frontend build: `lelab --dev` serves directly from Vite, and `lelab --rebuild` builds `frontend/dist/` with your local Node.js 22 when you want to try the production bundle. The SO-101 meshes in `frontend/public/` are kept out of LFS because installing from git does not fetch LFS files.

Run the Python tests with `pytest` (config in [pyproject.toml](pyproject.toml); install dev deps via `pip install -e ".[test]"`). Tests live in [tests/](tests/) and cover request schemas, pure helpers, and idle/mutex branches of the feature handlers — subprocess/thread happy paths and HF Jobs integration are deliberately not unit-tested. Lint with `ruff check` / `ruff format` (config in [pyproject.toml](pyproject.toml)). There is no Python build step; for end-to-end validation, run `lelab` and exercise endpoints (curl or via the frontend).

## Architecture

### Backend module layout (`lelab/`)

[server.py](lelab/server.py) is a thin FastAPI router. Each feature lives in its own module that owns its global state and exposes `handle_*` functions plus a Pydantic request model:

- [record.py](lelab/record.py): dataset recording around LeRobot's `record_loop`; frontend buttons set the same `events` dict LeRobot's keyboard listener would. The worker is a daemon thread, so the server's shutdown hook calls `stop_recording_and_wait()` to let an active session finalize on Ctrl+C.
- [teleoperate.py](lelab/teleoperate.py): leader→follower teleoperation loop over LeRobot's SO-101 leader and follower.
- [replay.py](lelab/replay.py): plays a recorded episode's actions on the follower arm, an in-process take on `lerobot-replay` so the dataset browser can stop it.
- [calibrate.py](lelab/calibrate.py) — step-by-step web calibration with a `CalibrationManager` singleton and `_step_complete` threading.Event.
- [train.py](lelab/train.py) — wraps the LeRobot training CLI as a subprocess (psutil for lifecycle, queue for log streaming).
- [datasets.py](lelab/datasets.py) — dataset listing (local cache + Hub, merged under a `source` field) plus the episode-browsing handlers behind `/dataset-episodes`, `/dataset-episode`, `/dataset-frame`, `/dataset-thumbnails`, `/dataset-motion` and `/dataset-video`.
- [episode_media.py](lelab/episode_media.py) — reads the LeRobot v3.0 on-disk layout (`meta/episodes/*.parquet` + `videos/`) and decodes frames with PyAV. Deliberately does **not** go through `LeRobotDataset`, so browsing a dataset never builds a robot config or imports torch. Several episodes share one mp4, so locating an episode resolves both the file and the `from_/to_timestamp` window inside it — which is why `/dataset-video` is addressed by chunk/file rather than by episode. Call `ensure_episode_index(repo_id)` before opening a local dataset with `LeRobotDataset`: LeRobot writes the index only in `finalize()`, and without it reads the dataset as "not downloaded yet" and fetches it from the Hub, which 404s for a dataset that was never pushed.
- [remote_ssh.py](lelab/remote_ssh.py) — in-memory paramiko sessions to the user's own servers (profiles saved by `utils/config.py`, never a password). Unknown host keys answer 409 with the fingerprint until the UI confirms it. Sessions carry SFTP and the `direct-tcpip` tunnel to a policy server; `lease()` marks one in use so it can't be disconnected mid-upload or mid-run.
- [remote_upload.py](lelab/remote_upload.py) — SFTP copy of a local dataset to `<remote_dir>/<name>`: `.part` + rename per file (a right-sized file is finished, so a repeat resumes), `meta/` last.
- [policy_server.py](lelab/policy_server.py) — **standalone** script (no `lelab` imports; torch/lerobot imported lazily) that serves a checkpoint over TCP on the GPU server, wire-compatible with the SO-101 sim project's policy server. Copied to the server by `/remote-servers/{name}/install-policy-server`; also the `lelab-policy-server` entry point. Keep it importable without LeLab (`tests/test_policy_server.py` checks).
- [remote_policy.py](lelab/remote_policy.py) — in-process inference on the follower with that remote policy: chunk queue with prefetch and stale-action skip, timeouts, `max_relative_target`, return to the start pose on any fault. Shares `rollout.inference_active` (via `_claim_inference_slot`) and `/inference-status` / `/stop-inference` with local runs (`rollout._inference_mode`).
- [run_video.py](lelab/run_video.py) — the optional mp4 of a remote run: one camera full frame, another (a wrist camera by default) inset bottom right. `add()` from the control loop only queues; a thread composes and encodes H.264 with PyAV, dropping frames rather than slowing the arm. Served by `/run-videos/{name}`.
- [utils/bus_guard.py](lelab/utils/bus_guard.py) — rides out brief USB glitches on an arm's bus: wraps `sync_read`/`sync_write` so a failed read returns the last good value and a failed write is skipped, and raises `ArmCommunicationLostError` once an arm has been silent for `MAX_OUTAGE_S` (10 s). Teleoperation, recording, replay and remote inference install it after device setup and remove it before the disconnect; local inference (`lerobot-rollout`) runs in its own process and isn't covered.
- [utils/config.py](lelab/utils/config.py) — shared paths and persistence: calibration JSON, saved ports, saved config selections. **Import shared constants from here, do not hardcode paths in feature modules.**

### State model

Each feature module owns module-level globals (e.g. `recording_active`, `teleoperation_active`, `current_robot`) protected by threads/locks where needed. There's no shared session object. Recording, teleoperation, inference and replay all drive the follower, so each start handler checks the others' flags and refuses to run while one is active. Calibration is not part of that check.

### WebSocket broadcast

[server.py](lelab/server.py#L104-L206) defines a single `ConnectionManager` with a background `_broadcast_worker` thread that drains a `queue.Queue` and forwards joint data to all `/ws/joint-data` clients via a thread-local asyncio loop. Feature modules get the manager passed in (e.g. `handle_start_teleoperation(request, manager)`) and call `manager.broadcast_joint_data_sync(data)` from their worker threads. Don't `await` from these threads — use the sync queue method.

### Persistent state on disk

All under `~/.cache/huggingface/lerobot/` (managed in [utils/config.py](lelab/utils/config.py)):

- `calibration/teleoperators/so101_leader/*.json` — leader calibration files (also called "teleop")
- `calibration/robots/so101_follower/*.json` — follower calibration files (also called "robot")
- `ports/{leader,follower}_port.txt` — last-used serial ports
- `saved_configs/{leader,follower}_config.txt` — last-selected config name
- `remote_servers/*.json` — remote SSH server profiles (no passwords); `remote_known_hosts` — host keys the user trusted in the UI
- `run_videos/*.mp4` — videos of remote model runs (`RUN_VIDEOS_PATH`)

`device_type` in API requests is `"teleop"` or `"robot"` (mapped to leader/follower paths). `robot_type` in port endpoints is `"leader"` or `"follower"`. Don't conflate the two vocabularies.

### Calibration files: dual-location pattern

[setup_calibration_files](lelab/utils/config.py#L30-L74) copies user-selected configs from `LEADER_CONFIG_PATH` / `FOLLOWER_CONFIG_PATH` into LeRobot's expected locations under `~/.cache/huggingface/lerobot/calibration/`. Recording, teleoperation, and replay all call this before starting. New features that drive a robot must do the same.

## Hardware target

Hardcoded for **SO-101 leader/follower arms** (`so101_leader`, `so101_follower`). Adding another robot type requires touching every feature module's config construction (search for `SO101LeaderConfig` / `SO101FollowerConfig`).

## Phone camera / HTTPS

Phone-as-camera streaming requires HTTPS. See [PHONE_CAMERA_SETUP.md](PHONE_CAMERA_SETUP.md) and [HTTPS_SETUP.md](HTTPS_SETUP.md). Self-signed certs go in `certs/`. The default `lelab-fullstack` does **not** start with HTTPS — for phone cameras run uvicorn manually with `--ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem`.

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Copy a recorded dataset to the user's own server over SFTP, so they can
train on it there.

The local folder (`~/.cache/huggingface/lerobot/<namespace>/<name>`) lands at
`<remote_dir>/<name>` with the same LeRobot v3 layout, ready for
`lerobot-train --dataset.root=<remote_dir>/<name>`. SFTP rather than scp, since
some cluster shells (Slurm Guard) refuse scp.

Resumable and safe to repeat: every file goes to `<file>.part` and is renamed
when complete, so a file with the right size on the server is a finished one
and is skipped next time. `meta/` goes last, so the copy only looks like a
dataset once everything else is there. One upload at a time; the UI polls the
status.

The same SFTP session also installs the Slurm training kit (lelab/slurm/) in the
server's work folder: `.env.example` (folders and training settings), the
`slurm/*.sbatch` jobs that train ACT and SmolVLA (full and LoRA) and serve a
trained policy, and the policy server script they run.
"""

from __future__ import annotations

import io
import logging
import posixpath
import shlex
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import paramiko
from pydantic import BaseModel

from . import episode_media, remote_ssh
from .utils import config

logger = logging.getLogger(__name__)


class RemoteUploadRequest(BaseModel):
    server: str
    dataset_repo_id: str
    # Folder on the server that receives the dataset folder. Relative paths and
    # `~/...` are under the login's home. Empty: the profile's remote_dataset_dir.
    remote_dir: str = ""
    # Name of the dataset folder on the server. Empty: the local folder's name.
    remote_name: str = ""
    # Also send the Slurm training kit to the profile's work folder, with DATASET_DIR set to remote_dir.
    send_training_kit: bool = False


class InstallPolicyServerRequest(BaseModel):
    # Folder on the server for the script. Empty: the profile's remote_work_dir, else the home.
    remote_dir: str = ""


class UploadCancelledError(Exception):
    pass


class TrainingKitRequest(BaseModel):
    # Project folder for .env.example and slurm/. Empty: the profile's remote_work_dir, else the home.
    remote_dir: str = ""
    # DATASET_DIR written into .env.example. Empty: the profile's remote_dataset_dir, else the home.
    dataset_dir: str = ""


POLICY_SERVER_FILE = "lelab_policy_server.py"
TRAINING_KIT_DIR = Path(__file__).with_name("slurm")
# Files of lelab/slurm/ that go to <project>/slurm/, in this order.
TRAINING_KIT_SCRIPTS = (
    "lelab_common.sh",
    "lelab_train.sh",
    "train_act.sbatch",
    "train_act_lora.sbatch",
    "train_smolvla_full.sbatch",
    "train_smolvla_lora.sbatch",
    "serve_policy.sbatch",
)
TRAINING_JOBS = tuple(f for f in TRAINING_KIT_SCRIPTS if f.startswith("train_"))


_lock = threading.Lock()
_thread: threading.Thread | None = None
_cancel = threading.Event()
_state: dict[str, Any] = {"state": "idle"}


def _plan_files(local_dir: Path) -> list[tuple[str, int]]:
    """(posix relative path, size) of every file to send, with `meta/` last.

    Dotfiles (.DS_Store and friends) are left out.
    """
    files = []
    for path in sorted(local_dir.rglob("*")):
        rel = path.relative_to(local_dir)
        if not path.is_file() or any(part.startswith(".") for part in rel.parts):
            continue
        files.append((rel.as_posix(), path.stat().st_size))
    return sorted(files, key=lambda f: (f[0].startswith("meta/"), f[0]))


def resolve_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str) -> str:
    """Absolute folder on the server. SFTP doesn't expand `~`; relative paths start at the home."""
    d = remote_dir.strip()
    if d in ("", "~"):
        d = ""
    elif d.startswith("~/"):
        d = d[2:]
    if not d.startswith("/"):
        d = posixpath.join(sftp.normalize("."), d)
    return posixpath.normpath(d)


def _check_name(name: str) -> str:
    name = name.strip()
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise remote_ssh.RemoteServerError(f"{name!r} can't be used as a folder name on the server.")
    return name


def _mkdirs(sftp: paramiko.SFTPClient, path: str, made: set[str]) -> None:
    """mkdir -p, remembering what exists so each folder costs one round trip."""
    if path in made or path in ("", "/"):
        return
    try:
        sftp.stat(path)
    except FileNotFoundError:
        _mkdirs(sftp, posixpath.dirname(path), made)
        sftp.mkdir(path)
    made.add(path)


def _remote_size(sftp: paramiko.SFTPClient, path: str) -> int | None:
    try:
        return sftp.stat(path).st_size
    except FileNotFoundError:
        return None


def _upload_files(
    sftp: paramiko.SFTPClient,
    local_dir: Path,
    remote_root: str,
    files: list[tuple[str, int]],
    progress: Callable[..., None],
    cancel: threading.Event,
) -> int:
    """Send `files` under `remote_root`. Returns how many were already there and skipped."""
    made: set[str] = set()
    skipped = 0
    for rel, size in files:
        if cancel.is_set():
            raise UploadCancelledError()
        remote = posixpath.join(remote_root, rel)
        progress(current_file=rel)
        _mkdirs(sftp, posixpath.dirname(remote), made)
        if _remote_size(sftp, remote) == size:
            skipped += 1
            progress(done_bytes=size, done_file=True)
            continue

        sent = 0

        def on_bytes(transferred: int, _total: int) -> None:
            nonlocal sent
            if cancel.is_set():
                raise UploadCancelledError()
            progress(done_bytes=transferred - sent)
            sent = transferred

        _put_file(sftp, local_dir / rel, remote, on_bytes)
        progress(done_bytes=size - sent, done_file=True)
    return skipped


def _put_file(sftp: paramiko.SFTPClient, local: Path, remote: str, callback=None) -> None:
    """Write to <remote>.part, then rename it into place, so a complete-looking file is a complete one."""
    tmp = remote + ".part"
    sftp.put(str(local), tmp, callback=callback)
    _move_into_place(sftp, tmp, remote)


def _put_bytes(sftp: paramiko.SFTPClient, data: bytes, remote: str, mode: int | None = None) -> None:
    """_put_file for generated content."""
    tmp = remote + ".part"
    sftp.putfo(io.BytesIO(data), tmp)
    if mode is not None:
        sftp.chmod(tmp, mode)
    _move_into_place(sftp, tmp, remote)


def _move_into_place(sftp: paramiko.SFTPClient, tmp: str, remote: str) -> None:
    try:
        sftp.posix_rename(tmp, remote)
    except OSError:
        # Servers without the posix-rename extension refuse to rename over a file.
        if _remote_size(sftp, remote) is not None:
            sftp.remove(remote)
        sftp.rename(tmp, remote)


def train_command(remote_root: str, name: str) -> str:
    """A starting point for training on the uploaded copy."""
    return (
        "lerobot-train "
        f"--dataset.repo_id={shlex.quote('local/' + name)} "
        f"--dataset.root={shlex.quote(remote_root)} "
        "--policy.type=act --policy.device=cuda --policy.push_to_hub=false "
        f"--output_dir={shlex.quote('outputs/act_' + name)}"
    )


def _sh_single_quoted(value: str) -> str:
    """`value` for the inside of a '...' shell string."""
    return value.replace("'", "'\\''")


def render_training_kit_file(name: str, project_dir: str, dataset_dir: str) -> str:
    """A kit file with this server's folders written in."""
    text = (TRAINING_KIT_DIR / name).read_text()
    text = text.replace("__LELAB_PROJECT_DIR__", _sh_single_quoted(project_dir))
    text = text.replace("{project}", project_dir)
    return text.replace("__LELAB_DATASET_DIR__", shlex.quote(dataset_dir))


def slurm_train_commands(project_dir: str, dataset_path: str) -> list[str]:
    """One sbatch line per training job, for a dataset at `dataset_path` on the server."""
    return [
        f"DATASET={shlex.quote(dataset_path)} sbatch {shlex.quote(posixpath.join(project_dir, 'slurm', job))}"
        for job in TRAINING_JOBS
    ]


def install_training_kit(
    sftp: paramiko.SFTPClient, profile: dict[str, Any], remote_dir: str = "", dataset_dir: str = ""
) -> dict[str, Any]:
    """Copy .env.example, slurm/ and the policy server to the project folder.

    Scripts and .env.example are replaced every time, so a newer LeLab updates
    them; .env holds the user's own settings and is only written when missing.
    """
    project = resolve_remote_dir(sftp, remote_dir or profile.get("remote_work_dir", ""))
    datasets = resolve_remote_dir(sftp, dataset_dir or profile.get("remote_dataset_dir", ""))
    slurm_dir = posixpath.join(project, "slurm")
    _mkdirs(sftp, slurm_dir, set())
    files = []
    for name in TRAINING_KIT_SCRIPTS:
        remote = posixpath.join(slurm_dir, name)
        _put_bytes(sftp, render_training_kit_file(name, project, datasets).encode(), remote, 0o755)
        files.append(remote)
    server = posixpath.join(project, POLICY_SERVER_FILE)
    _put_file(sftp, Path(__file__).with_name("policy_server.py"), server)
    files.append(server)
    env_text = render_training_kit_file("env.example", project, datasets).encode()
    _put_bytes(sftp, env_text, posixpath.join(project, ".env.example"))
    files.append(posixpath.join(project, ".env.example"))
    env_path = posixpath.join(project, ".env")
    created_env = _remote_size(sftp, env_path) is None
    if created_env:
        _put_bytes(sftp, env_text, env_path, 0o600)  # it may hold HF_TOKEN
        files.append(env_path)
    return {
        "remote_dir": project,
        "dataset_dir": datasets,
        "files": files,
        "created_env": created_env,
        "env_path": env_path,
        "serve_command": f"sbatch {shlex.quote(posixpath.join(slurm_dir, 'serve_policy.sbatch'))}",
    }


def _has_training_kit(sftp: paramiko.SFTPClient, profile: dict[str, Any]) -> str | None:
    """The project folder, when the kit is already installed there."""
    try:
        project = resolve_remote_dir(sftp, profile.get("remote_work_dir", ""))
        return project if _remote_size(sftp, posixpath.join(project, "slurm", "lelab_train.sh")) else None
    except (OSError, paramiko.SSHException):
        return None


def _set(**fields: Any) -> None:
    with _lock:
        _state.update(fields)


def _worker(request: RemoteUploadRequest, local_dir: Path, name: str, files: list[tuple[str, int]]) -> None:
    def progress(current_file: str | None = None, done_bytes: int = 0, done_file: bool = False) -> None:
        with _lock:
            if current_file is not None:
                _state["current_file"] = current_file
            _state["bytes_done"] += done_bytes
            if done_file:
                _state["files_done"] += 1

    try:
        with remote_ssh.lease(request.server) as session:
            sftp = remote_ssh.open_sftp(session)
            try:
                remote_dir = request.remote_dir or session.profile.get("remote_dataset_dir", "")
                dataset_dir = resolve_remote_dir(sftp, remote_dir)
                remote_root = posixpath.join(dataset_dir, name)
                _set(remote_path=remote_root)
                skipped = _upload_files(sftp, local_dir, remote_root, files, progress, _cancel)
                kit: dict[str, Any] | None = None
                kit_error = None
                if request.send_training_kit:
                    _set(current_file="Slurm training files")
                    try:
                        kit = install_training_kit(sftp, session.profile, dataset_dir=dataset_dir)
                    except (OSError, paramiko.SSHException) as exc:
                        # The dataset is there; only the extras failed.
                        logger.warning("Could not send the Slurm training files: %s", exc)
                        kit_error = f"The dataset arrived, but the Slurm files could not be sent: {exc}"
                project = kit["remote_dir"] if kit else _has_training_kit(sftp, session.profile)
            finally:
                sftp.close()
        _set(
            state="done",
            files_skipped=skipped,
            current_file=None,
            finished_at=time.time(),
            train_command=train_command(remote_root, name),
            slurm_commands=slurm_train_commands(project, remote_root) if project else None,
            training_kit=kit,
            training_kit_error=kit_error,
        )
        logger.info(
            "Sent %s to %s:%s (%d files already there)",
            request.dataset_repo_id,
            request.server,
            remote_root,
            skipped,
        )
    except UploadCancelledError:
        _set(state="cancelled", current_file=None, finished_at=time.time())
        logger.info("Upload of %s cancelled", request.dataset_repo_id)
    except remote_ssh.RemoteServerError as exc:
        _set(state="error", error=exc.message, finished_at=time.time())
    except Exception as exc:
        logger.exception("Upload of %s to %s failed", request.dataset_repo_id, request.server)
        _set(state="error", error=f"Upload failed: {exc}", finished_at=time.time())


def handle_start_upload(request: RemoteUploadRequest) -> dict[str, Any]:
    """Validate, then copy in the background. Raises RemoteServerError for the route to report."""
    global _thread

    from . import record as _record

    if config.get_remote_server(request.server) is None:
        raise remote_ssh.RemoteServerError(f"No saved server named {request.server!r}", 404)
    local_dir = _record._resolve_dataset_dir(request.dataset_repo_id)
    if local_dir is None or not (local_dir / "meta" / "info.json").is_file():
        raise remote_ssh.RemoteServerError(f"Dataset {request.dataset_repo_id} not found locally", 404)
    recording = _record.recording_config
    if (
        _record.recording_active
        and recording is not None
        and recording.dataset_repo_id == request.dataset_repo_id
    ):
        raise remote_ssh.RemoteServerError(
            "This dataset is still being recorded. Stop the recording first.", 409
        )
    try:
        episode_media.ensure_episode_index(request.dataset_repo_id)
    except episode_media.UnreadableDatasetError as exc:
        raise remote_ssh.RemoteServerError(str(exc), 409) from None
    name = _check_name(request.remote_name or local_dir.name)
    remote_ssh.ensure_connected(request.server)

    files = _plan_files(local_dir)
    with _lock:
        if _thread is not None and _thread.is_alive():
            raise remote_ssh.RemoteServerError("Another upload is running. Wait for it or cancel it.", 409)
        _cancel.clear()
        _state.clear()
        _state.update(
            state="running",
            server=request.server,
            dataset_repo_id=request.dataset_repo_id,
            remote_name=name,
            remote_path=None,
            bytes_total=sum(size for _, size in files),
            bytes_done=0,
            files_total=len(files),
            files_done=0,
            files_skipped=0,
            current_file=None,
            error=None,
            train_command=None,
            slurm_commands=None,
            training_kit=None,
            training_kit_error=None,
            started_at=time.time(),
            finished_at=None,
        )
        _thread = threading.Thread(
            target=_worker, args=(request, local_dir, name, files), name="remote-upload", daemon=True
        )
        _thread.start()
    return handle_upload_status()


def handle_upload_status() -> dict[str, Any]:
    with _lock:
        return dict(_state)


def handle_cancel_upload() -> dict[str, Any]:
    with _lock:
        running = _thread is not None and _thread.is_alive()
    if not running:
        raise remote_ssh.RemoteServerError("No upload is running.", 409)
    _cancel.set()
    return {"success": True}


def cancel_and_wait(timeout: float = 5.0) -> None:
    """Stop a running upload (server shutdown). Finished files stay; the next upload resumes."""
    thread = _thread
    if thread is not None and thread.is_alive():
        _cancel.set()
        thread.join(timeout)


def handle_install_policy_server(name: str, request: InstallPolicyServerRequest) -> dict[str, Any]:
    """Copy lelab/policy_server.py to the server, where it runs with the Python that trains there."""
    if config.get_remote_server(name) is None:
        raise remote_ssh.RemoteServerError(f"No saved server named {name!r}", 404)
    with remote_ssh.lease(name) as session:
        sftp = remote_ssh.open_sftp(session)
        try:
            remote_dir = resolve_remote_dir(
                sftp, request.remote_dir or session.profile.get("remote_work_dir", "")
            )
            _mkdirs(sftp, remote_dir, set())
            remote_path = posixpath.join(remote_dir, POLICY_SERVER_FILE)
            _put_file(sftp, Path(__file__).with_name("policy_server.py"), remote_path)
        finally:
            sftp.close()
    logger.info("Copied the policy server to %s:%s", name, remote_path)
    return {
        "remote_path": remote_path,
        "command": (
            f"cd {shlex.quote(remote_dir)} && python {POLICY_SERVER_FILE} "
            "--policy <run folder>/checkpoints/last/pretrained_model --port 5555"
        ),
    }


def handle_install_training_kit(name: str, request: TrainingKitRequest) -> dict[str, Any]:
    """Send the Slurm training kit (lelab/slurm/ and the policy server) to the server's project folder."""
    if config.get_remote_server(name) is None:
        raise remote_ssh.RemoteServerError(f"No saved server named {name!r}", 404)
    with remote_ssh.lease(name) as session:
        sftp = remote_ssh.open_sftp(session)
        try:
            kit = install_training_kit(sftp, session.profile, request.remote_dir, request.dataset_dir)
        except (OSError, paramiko.SSHException) as exc:
            raise remote_ssh.RemoteServerError(f"Could not send the Slurm files: {exc}", 502) from None
        finally:
            sftp.close()
    logger.info("Sent the Slurm training files to %s:%s", name, kit["remote_dir"])
    return kit

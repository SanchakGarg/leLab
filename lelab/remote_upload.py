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
"""

from __future__ import annotations

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


class UploadCancelledError(Exception):
    pass


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

        tmp = remote + ".part"
        sftp.put(str(local_dir / rel), tmp, callback=on_bytes)
        try:
            sftp.posix_rename(tmp, remote)
        except OSError:
            # Servers without the posix-rename extension refuse to rename over a file.
            if _remote_size(sftp, remote) is not None:
                sftp.remove(remote)
            sftp.rename(tmp, remote)
        progress(done_bytes=size - sent, done_file=True)
    return skipped


def train_command(remote_root: str, name: str) -> str:
    """A starting point for training on the uploaded copy."""
    return (
        "lerobot-train "
        f"--dataset.repo_id={shlex.quote('local/' + name)} "
        f"--dataset.root={shlex.quote(remote_root)} "
        "--policy.type=act --policy.device=cuda --policy.push_to_hub=false "
        f"--output_dir={shlex.quote('outputs/act_' + name)}"
    )


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
                remote_root = posixpath.join(resolve_remote_dir(sftp, remote_dir), name)
                _set(remote_path=remote_root)
                skipped = _upload_files(sftp, local_dir, remote_root, files, progress, _cancel)
            finally:
                sftp.close()
        _set(
            state="done",
            files_skipped=skipped,
            current_file=None,
            finished_at=time.time(),
            train_command=train_command(remote_root, name),
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

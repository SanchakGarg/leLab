"""Tests for lelab.remote_upload: what is sent, in which order, resume and
cancel, and the checks before an upload starts. The SFTP server is a fake that
maps remote paths into a tmp folder."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lelab.policy_server import LELAB_KIT_VERSION

HOME = "/raid/home/student"


class FakeSftp:
    """The SFTPClient calls remote_upload makes, on a local folder."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.puts: list[str] = []
        (root / HOME.lstrip("/")).mkdir(parents=True, exist_ok=True)

    def _p(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    def normalize(self, path: str) -> str:
        return HOME

    def stat(self, path: str) -> os.stat_result:
        p = self._p(path)
        if not p.exists():
            raise FileNotFoundError(path)
        return p.stat()

    def mkdir(self, path: str) -> None:
        p = self._p(path)
        if not p.parent.exists():
            raise FileNotFoundError(path)
        p.mkdir()

    def put(self, local: str, remote: str, callback=None) -> None:
        self.puts.append(remote)
        data = Path(local).read_bytes()
        with open(self._p(remote), "wb") as f:
            for i in range(0, max(len(data), 1), 4):
                f.write(data[i : i + 4])
                if callback is not None:
                    callback(min(i + 4, len(data)), len(data))

    def putfo(self, fl, remote: str) -> None:
        self.puts.append(remote)
        self._p(remote).write_bytes(fl.read())

    def open(self, path: str, mode: str = "r"):
        assert mode == "rb", mode
        return open(self._p(path), "rb")

    def chmod(self, path: str, mode: int) -> None:
        self._p(path).chmod(mode)

    def posix_rename(self, old: str, new: str) -> None:
        os.replace(self._p(old), self._p(new))

    def rename(self, old: str, new: str) -> None:
        os.rename(self._p(old), self._p(new))

    def remove(self, path: str) -> None:
        self._p(path).unlink()

    def close(self) -> None:
        pass


def _make_dataset(root: Path) -> Path:
    ds = root / "local" / "pick_cube_20261117_120000"
    files = {
        "meta/info.json": '{"fps": 30}',
        "meta/episodes/chunk-000/file-000.parquet": "episodes",
        "data/chunk-000/file-000.parquet": "frames" * 10,
        "videos/observation.images.front/chunk-000/file-000.mp4": "video" * 20,
        ".DS_Store": "junk",
    }
    for rel, content in files.items():
        path = ds / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return ds


@pytest.fixture
def dataset(tmp_lerobot_home: Path) -> Path:
    return _make_dataset(tmp_lerobot_home)


@pytest.fixture
def sftp(tmp_path: Path) -> FakeSftp:
    return FakeSftp(tmp_path / "server")


def test_plan_sends_meta_last_and_skips_dotfiles(dataset: Path) -> None:
    from lelab.remote_upload import _plan_files

    names = [rel for rel, _ in _plan_files(dataset)]

    assert ".DS_Store" not in names
    assert names[-2:] == ["meta/episodes/chunk-000/file-000.parquet", "meta/info.json"]
    assert set(names[:-2]) == {
        "data/chunk-000/file-000.parquet",
        "videos/observation.images.front/chunk-000/file-000.mp4",
    }


@pytest.mark.parametrize(
    ("remote_dir", "expected"),
    [
        ("", HOME),
        ("~", HOME),
        ("~/projects/data", f"{HOME}/projects/data"),
        ("projects/data/", f"{HOME}/projects/data"),
        ("/scratch/student", "/scratch/student"),
    ],
)
def test_remote_dir_resolution(sftp: FakeSftp, remote_dir: str, expected: str) -> None:
    from lelab.remote_upload import resolve_remote_dir

    assert resolve_remote_dir(sftp, remote_dir) == expected


def test_upload_copies_every_file_and_renames_parts(dataset: Path, sftp: FakeSftp) -> None:
    from lelab.remote_upload import _plan_files, _upload_files

    files = _plan_files(dataset)
    remote_root = f"{HOME}/projects/data/{dataset.name}"
    skipped = _upload_files(sftp, dataset, remote_root, files, lambda **_: None, threading.Event())

    assert skipped == 0
    copied = sftp._p(remote_root)
    for rel, _ in files:
        assert (copied / rel).read_bytes() == (dataset / rel).read_bytes()
    assert not list(copied.rglob("*.part"))
    assert all(p.endswith(".part") for p in sftp.puts)
    assert sftp.puts[-1].endswith("meta/info.json.part")


def test_a_second_upload_skips_finished_files(dataset: Path, sftp: FakeSftp) -> None:
    from lelab.remote_upload import _plan_files, _upload_files

    files = _plan_files(dataset)
    remote_root = f"{HOME}/{dataset.name}"
    _upload_files(sftp, dataset, remote_root, files, lambda **_: None, threading.Event())
    # A half-sent file from an interrupted run has the wrong size and is sent again.
    (sftp._p(remote_root) / "data/chunk-000/file-000.parquet").write_text("half")
    sftp.puts.clear()

    skipped = _upload_files(sftp, dataset, remote_root, files, lambda **_: None, threading.Event())

    assert skipped == len(files) - 1
    assert sftp.puts == [f"{remote_root}/data/chunk-000/file-000.parquet.part"]


def test_progress_adds_up_to_the_total(dataset: Path, sftp: FakeSftp) -> None:
    from lelab.remote_upload import _plan_files, _upload_files

    files = _plan_files(dataset)
    totals = {"bytes": 0, "files": 0}

    def progress(current_file=None, done_bytes=0, done_file=False) -> None:
        totals["bytes"] += done_bytes
        totals["files"] += int(done_file)

    _upload_files(sftp, dataset, f"{HOME}/x", files, progress, threading.Event())

    assert totals == {"bytes": sum(size for _, size in files), "files": len(files)}


def test_cancel_stops_before_meta(dataset: Path, sftp: FakeSftp) -> None:
    from lelab.remote_upload import UploadCancelledError, _plan_files, _upload_files

    files = _plan_files(dataset)
    cancel = threading.Event()

    def progress(current_file=None, done_bytes=0, done_file=False) -> None:
        if done_file:
            cancel.set()

    with pytest.raises(UploadCancelledError):
        _upload_files(sftp, dataset, f"{HOME}/x", files, progress, cancel)
    assert not (sftp._p(f"{HOME}/x") / "meta" / "info.json").exists()


@pytest.fixture
def connected(tmp_lerobot_home: Path, sftp: FakeSftp, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A saved, connected server whose SFTP is the fake."""
    from lelab import episode_media, remote_ssh, remote_upload
    from lelab.utils import config as cfg

    profile = cfg.save_remote_server(
        "cluster",
        {"host": "10.2.100.14", "username": "student", "auth": "password", "remote_dataset_dir": "data"},
    )
    session = SimpleNamespace(name="cluster", profile=profile, users=0)
    monkeypatch.setattr(remote_ssh, "ensure_connected", lambda name: session)
    monkeypatch.setattr(remote_ssh, "open_sftp", lambda s: sftp)
    monkeypatch.setattr(episode_media, "ensure_episode_index", lambda repo_id: None)
    monkeypatch.setattr(remote_upload, "_state", {"state": "idle"})
    monkeypatch.setattr(remote_upload, "_thread", None)
    return session


def _wait(remote_upload) -> dict:
    thread = remote_upload._thread
    assert thread is not None
    thread.join(5)
    return remote_upload.handle_upload_status()


def test_upload_runs_in_the_background(dataset: Path, sftp: FakeSftp, connected: SimpleNamespace) -> None:
    from lelab import remote_upload

    started = remote_upload.handle_start_upload(
        remote_upload.RemoteUploadRequest(server="cluster", dataset_repo_id=f"local/{dataset.name}")
    )
    assert started["state"] == "running"

    status = _wait(remote_upload)
    assert status["state"] == "done", status
    assert status["remote_path"] == f"{HOME}/data/{dataset.name}"
    assert status["bytes_done"] == status["bytes_total"]
    assert status["files_done"] == status["files_total"] == 4
    assert f"--dataset.root={HOME}/data/{dataset.name}" in status["train_command"]
    assert (sftp._p(status["remote_path"]) / "meta" / "info.json").is_file()
    assert connected.users == 0


def test_remote_name_overrides_the_folder_name(
    dataset: Path, sftp: FakeSftp, connected: SimpleNamespace
) -> None:
    from lelab import remote_upload

    remote_upload.handle_start_upload(
        remote_upload.RemoteUploadRequest(
            server="cluster",
            dataset_repo_id=f"local/{dataset.name}",
            remote_dir="/scratch",
            remote_name="so101_cube",
        )
    )
    assert _wait(remote_upload)["remote_path"] == "/scratch/so101_cube"


@pytest.mark.parametrize(
    ("overrides", "status_code"),
    [
        ({"server": "nope"}, 404),
        ({"dataset_repo_id": "local/missing"}, 404),
        ({"dataset_repo_id": "../../etc"}, 404),
        ({"remote_name": "../up"}, 400),
    ],
)
def test_bad_requests_are_refused(
    dataset: Path, connected: SimpleNamespace, overrides: dict, status_code: int
) -> None:
    from lelab import remote_ssh, remote_upload

    body = {"server": "cluster", "dataset_repo_id": f"local/{dataset.name}", **overrides}
    with pytest.raises(remote_ssh.RemoteServerError) as exc:
        remote_upload.handle_start_upload(remote_upload.RemoteUploadRequest(**body))
    assert exc.value.status_code == status_code
    assert remote_upload._thread is None


def test_a_dataset_being_recorded_is_refused(
    dataset: Path, connected: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lelab import record, remote_ssh, remote_upload

    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_config", SimpleNamespace(dataset_repo_id=f"local/{dataset.name}"))

    with pytest.raises(remote_ssh.RemoteServerError) as exc:
        remote_upload.handle_start_upload(
            remote_upload.RemoteUploadRequest(server="cluster", dataset_repo_id=f"local/{dataset.name}")
        )
    assert exc.value.status_code == 409


def test_routes(dataset: Path, connected: SimpleNamespace, client: TestClient) -> None:
    from lelab import remote_upload

    assert client.post("/remote-upload/cancel").status_code == 409
    response = client.post(
        "/remote-upload", json={"server": "cluster", "dataset_repo_id": f"local/{dataset.name}"}
    )
    assert response.status_code == 200
    _wait(remote_upload)
    assert client.get("/remote-upload/status").json()["state"] == "done"


def test_unknown_dataset_route_is_404(connected: SimpleNamespace, client: TestClient) -> None:
    response = client.post("/remote-upload", json={"server": "cluster", "dataset_repo_id": "local/missing"})
    assert response.status_code == 404


def test_rename_fallback_replaces_an_existing_file(dataset: Path, sftp: FakeSftp) -> None:
    """Servers without posix-rename refuse to rename over a file; the old one is removed first."""
    from lelab.remote_upload import _plan_files, _upload_files

    def no_posix_rename(old: str, new: str) -> None:
        raise OSError("unsupported")

    sftp.posix_rename = no_posix_rename
    files = _plan_files(dataset)
    remote_root = f"{HOME}/x"
    (sftp._p(remote_root) / "meta").mkdir(parents=True)
    (sftp._p(remote_root) / "meta" / "info.json").write_text("stale, wrong size")

    _upload_files(sftp, dataset, remote_root, files, lambda **_: None, threading.Event())

    assert (sftp._p(remote_root) / "meta" / "info.json").read_text() == '{"fps": 30}'


def test_training_kit_lands_in_the_work_folder_with_its_folders_written_in(sftp: FakeSftp) -> None:
    from lelab import remote_upload

    profile = {"remote_work_dir": "projects/lelab", "remote_dataset_dir": "data"}
    kit = remote_upload.install_training_kit(sftp, profile)

    project = f"{HOME}/projects/lelab"
    assert kit["remote_dir"] == project and kit["created_env"] is True
    root = sftp._p(project)
    for name in remote_upload.TRAINING_KIT_SCRIPTS:
        script = root / "slurm" / name
        assert script.is_file() and os.access(script, os.X_OK), name
        assert "__LELAB" not in script.read_text().replace("__LELAB*)", "")
    assert (
        f"PROJECT_DIR=${{LELAB_PROJECT_DIR:-'{project}'}}"
        in (root / "slurm" / "train_act.sbatch").read_text()
    )
    assert f"DATASET_DIR={HOME}/data\n" in (root / ".env.example").read_text()
    assert (root / ".env").read_text() == (root / ".env.example").read_text()
    assert (root / ".env").stat().st_mode & 0o777 == 0o600
    assert (root / remote_upload.POLICY_SERVER_FILE).is_file()
    assert not list(root.rglob("*.part"))


def test_training_kit_never_overwrites_the_users_env(sftp: FakeSftp) -> None:
    from lelab import remote_upload

    kit = remote_upload.install_training_kit(sftp, {})
    env = sftp._p(kit["env_path"])
    env.write_text("DATASET=mine\n")

    again = remote_upload.install_training_kit(sftp, {}, dataset_dir="/scratch/data")

    assert again["created_env"] is False
    assert env.read_text() == "DATASET=mine\n"
    assert "DATASET_DIR=/scratch/data" in sftp._p(f"{HOME}/.env.example").read_text()


def test_training_kit_copies_only_missing_or_changed_files(sftp: FakeSftp) -> None:
    from lelab import remote_upload

    first = remote_upload.install_training_kit(sftp, {})
    statuses = {c["name"]: c["status"] for c in first["checks"]}
    assert set(statuses.values()) == {"added"}
    assert {
        "slurm/train_pi05_full.sbatch",
        "slurm/train_pi05_lora.sbatch",
        ".env.example",
        ".env",
    } <= statuses.keys()

    slurm = sftp._p(f"{HOME}/slurm")
    (slurm / "train_act.sbatch").write_text("an old version")
    (slurm / "train_pi05_lora.sbatch").unlink()
    (slurm / "lelab_train.sh").chmod(0o644)
    sftp.puts.clear()

    again = remote_upload.install_training_kit(sftp, {})

    statuses = {c["name"]: c["status"] for c in again["checks"]}
    assert statuses["slurm/train_act.sbatch"] == "updated"
    by_name = {c["name"]: c for c in again["checks"]}
    assert again["version"] == LELAB_KIT_VERSION
    assert by_name["slurm/train_act.sbatch"]["previous_version"] is None  # "an old version" had none
    assert by_name["slurm/lelab_train.sh"]["previous_version"] == LELAB_KIT_VERSION
    assert by_name["lelab_policy_server.py"]["version"] == LELAB_KIT_VERSION
    assert statuses["slurm/train_pi05_lora.sbatch"] == "added"
    assert statuses["slurm/lelab_train.sh"] == "up_to_date"
    assert os.access(slurm / "lelab_train.sh", os.X_OK)  # its mode was fixed in place
    assert statuses[".env"] == "kept"
    assert "#!/bin/bash" in (slurm / "train_act.sbatch").read_text()
    # Only the two changed files and .env.example (always rewritten) were sent.
    assert sorted(p.rsplit("/", 1)[1] for p in sftp.puts) == [
        ".env.example.part",
        "train_act.sbatch.part",
        "train_pi05_lora.sbatch.part",
    ]
    assert sorted(again["files"]) == [
        f"{HOME}/slurm/train_act.sbatch",
        f"{HOME}/slurm/train_pi05_lora.sbatch",
    ]


def test_training_kit_lists_the_settings_an_existing_env_lacks(sftp: FakeSftp) -> None:
    from lelab import remote_upload

    kit = remote_upload.install_training_kit(sftp, {})
    assert kit["env_missing_keys"] == []
    env = sftp._p(kit["env_path"])
    env.write_text(
        "".join(line + "\n" for line in env.read_text().splitlines() if not line.startswith("PI05_"))
    )

    again = remote_upload.install_training_kit(sftp, {})

    assert again["env_missing_keys"][:2] == ["PI05_BASE", "PI05_STEPS"]
    assert all(k.startswith("PI05_") for k in again["env_missing_keys"])
    assert "PI05_BASE=" not in env.read_text()  # still the user's file


def test_upload_can_send_the_training_kit(dataset: Path, sftp: FakeSftp, connected: SimpleNamespace) -> None:
    from lelab import remote_upload

    remote_upload.handle_start_upload(
        remote_upload.RemoteUploadRequest(
            server="cluster", dataset_repo_id=f"local/{dataset.name}", send_training_kit=True
        )
    )
    status = _wait(remote_upload)

    assert status["state"] == "done", status
    assert status["training_kit"]["dataset_dir"] == f"{HOME}/data"
    assert status["slurm_commands"][0] == (
        f"DATASET={HOME}/data/{dataset.name} sbatch {HOME}/slurm/train_act.sbatch"
    )
    assert len(status["slurm_commands"]) == len(remote_upload.TRAINING_JOBS)


def test_install_training_kit_route(connected: SimpleNamespace, client: TestClient) -> None:
    response = client.post("/remote-servers/cluster/install-training-kit", json={"remote_dir": "work"})
    assert response.status_code == 200, response.text
    assert response.json()["remote_dir"] == f"{HOME}/work"
    assert client.post("/remote-servers/nope/install-training-kit", json={}).status_code == 404


def test_kit_version_is_read_from_scripts_and_the_policy_server() -> None:
    from lelab.remote_upload import kit_version_of

    assert kit_version_of(b"#!/bin/bash\n# lelab-kit-version: 2026.11.18-1\n") == "2026.11.18-1"
    assert kit_version_of(b'PROTOCOL = 1\nLELAB_KIT_VERSION = "2026.11.18-2"\n') == "2026.11.18-2"
    assert kit_version_of(b"#!/bin/bash\necho hi\n") is None

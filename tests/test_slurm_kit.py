"""The Slurm training kit (lelab/slurm/) run for real with bash, against a fake
environment whose lerobot-train records its argv. The argv then goes through
the installed LeRobot's own parser and validation, like test_lerobot_cli.py
does for the training LeLab runs itself, so a renamed flag fails here."""

# No `from __future__ import annotations`: parser.wrap reads the config class off the annotation.
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import lerobot.scripts.lerobot_train  # noqa: F401  registers the policy configs, as the CLI does
from lelab import remote_upload
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

FAKE_TRAIN = """#!{python}
import json, os, pathlib, sys
pathlib.Path(os.environ["ARGV_OUT"]).write_text(json.dumps(sys.argv[1:]))
out = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--output_dir=")]
if out:
    ckpt = pathlib.Path(out[0]) / "checkpoints" / "000010" / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text("{{}}")
    (ckpt / "train_config.json").write_text("{{}}")
    os.symlink("000010", pathlib.Path(out[0]) / "checkpoints" / "last")
"""


@parser.wrap()
def _parse_training(cfg: TrainPipelineConfig) -> TrainPipelineConfig:
    return cfg


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A server project folder as LeLab installs it, a dataset, and a fake LeRobot environment."""
    proj, data = tmp_path / "proj", tmp_path / "data"
    (proj / "slurm").mkdir(parents=True)
    for name in remote_upload.TRAINING_KIT_SCRIPTS:
        (proj / "slurm" / name).write_text(remote_upload.render_training_kit_file(name, str(proj), str(data)))
    (proj / ".env").write_text(remote_upload.render_training_kit_file("env.example", str(proj), str(data)))

    meta = data / "pick_cube" / "meta"
    meta.mkdir(parents=True)
    features = {"observation.images.front": {}, "observation.images.wrist": {}, "observation.state": {}}
    (meta / "info.json").write_text(json.dumps({"features": features}))
    _write_frames(data / "pick_cube")

    bin_dir = proj / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    # `python -c ...` is only used for the torch/peft checks: pass them. Scripts on stdin run for real.
    (bin_dir / "python").write_text(
        f'#!/bin/bash\nif [ "$1" = -c ]; then exit 0; fi\nexec {sys.executable} "$@"\n'
    )
    (bin_dir / "lerobot-train").write_text(FAKE_TRAIN.format(python=sys.executable))
    for f in bin_dir.iterdir():
        f.chmod(0o755)

    SmolVLAConfig().save_pretrained(tmp_path / "smolvla_base")
    ACTConfig().save_pretrained(tmp_path / "act_init")
    # pi05_base's cameras and padded state, as on the Hub.
    image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
    cams = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    PI05Config(
        input_features={
            **{f"observation.images.{c}": image for c in cams},
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(32,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(32,))},
    ).save_pretrained(tmp_path / "pi05_base")
    return proj


def _write_frames(dataset: Path) -> None:
    """data/ with 100 frames of 6 joints, and stats.json without quantiles (a dataset from before LeRobot 0.7)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [[float(i + j) for j in range(6)] for i in range(100)]
    (dataset / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.table({"action": rows, "observation.state": rows, "index": list(range(100))}),
        dataset / "data" / "chunk-000" / "file-000.parquet",
    )
    stats = {k: {"mean": [0.0] * 6, "std": [1.0] * 6} for k in ("action", "observation.state")}
    (dataset / "meta" / "stats.json").write_text(json.dumps(stats))


def _run(project: Path, script: str, cwd: Path | None = None, **env: str) -> subprocess.CompletedProcess:
    full_env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "LELAB_"))}
    full_env.update(ARGV_OUT=str(project / "argv.json"), **env)
    return subprocess.run(
        ["bash", str(project / "slurm" / script)],
        cwd=cwd or project.parent,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize(
    ("script", "policy_type", "lora"),
    [
        ("train_act.sbatch", "act", False),
        ("train_act_lora.sbatch", "act", True),
        ("train_smolvla_full.sbatch", "smolvla", False),
        ("train_smolvla_lora.sbatch", "smolvla", True),
        ("train_pi05_full.sbatch", "pi05", False),
        ("train_pi05_lora.sbatch", "pi05", True),
    ],
)
def test_training_jobs_build_argv_lerobot_accepts(
    project: Path, monkeypatch: pytest.MonkeyPatch, script: str, policy_type: str, lora: bool
) -> None:
    tmp = project.parent
    result = _run(
        project,
        script,
        DATASET="pick_cube",
        SMOLVLA_BASE=str(tmp / "smolvla_base"),
        PI05_BASE=str(tmp / "pi05_base"),
        ACT_INIT=str(tmp / "act_init") if lora else "",
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    argv = json.loads((project / "argv.json").read_text())
    run = f"{policy_type}_{'lora' if lora else 'full'}_pick_cube"
    assert f"--output_dir={project}/outputs/{run}" in argv
    assert f"--dataset.root={tmp}/data/pick_cube" in argv
    # The finished policy is published for serve_policy.sbatch: a link to the real step folder.
    published = project / "policies" / run
    assert published.is_symlink() and (published / "config.json").is_file()
    assert published.resolve() == (project / "outputs" / run / "checkpoints" / "000010" / "pretrained_model")

    shutil.rmtree(project / "outputs")  # validate() refuses an existing output_dir on a new run
    monkeypatch.setattr(sys, "argv", ["lerobot-train", *argv])
    cfg = _parse_training()
    cfg.validate()
    assert cfg.policy.type == policy_type
    assert (cfg.peft is not None) == lora
    assert cfg.trackio.enable and cfg.trackio.project == "lelab"
    if policy_type == "smolvla":
        assert cfg.rename_map == {
            "observation.images.front": "observation.images.camera1",
            "observation.images.wrist": "observation.images.camera2",
        }
    if policy_type == "pi05":
        assert cfg.rename_map == {
            "observation.images.front": "observation.images.base_0_rgb",
            "observation.images.wrist": "observation.images.left_wrist_0_rgb",
        }
        assert cfg.policy.dtype == torch.bfloat16 and cfg.policy.gradient_checkpointing
        assert cfg.policy.train_expert_only == (not lora) and cfg.policy.scheduler_decay_steps == 30000


def test_pi05_puts_the_wrist_camera_in_a_wrist_slot_and_honours_pi05_cameras(project: Path) -> None:
    info = project.parent / "data" / "pick_cube" / "meta" / "info.json"
    features = {"observation.images.wrist": {}, "observation.images.top": {}, "observation.state": {}}
    info.write_text(json.dumps({"features": features}))
    tmp = project.parent

    result = _run(project, "train_pi05_lora.sbatch", DATASET="pick_cube", PI05_BASE=str(tmp / "pi05_base"))
    assert result.returncode == 0, result.stderr
    argv = json.loads((project / "argv.json").read_text())
    rename = json.loads(next(a for a in argv if a.startswith("--rename_map=")).split("=", 1)[1])
    assert rename == {
        "observation.images.top": "observation.images.base_0_rgb",
        "observation.images.wrist": "observation.images.left_wrist_0_rgb",
    }

    shutil.rmtree(project / "outputs")
    result = _run(
        project,
        "train_pi05_lora.sbatch",
        DATASET="pick_cube",
        PI05_BASE=str(tmp / "pi05_base"),
        PI05_CAMERAS="nope",
    )
    assert result.returncode == 1 and "PI05_CAMERAS names cameras" in result.stderr


def test_pi05_adds_the_quantile_stats_an_older_dataset_lacks(project: Path) -> None:
    meta = project.parent / "data" / "pick_cube" / "meta"
    original = (meta / "stats.json").read_text()

    result = _run(
        project, "train_pi05_lora.sbatch", DATASET="pick_cube", PI05_BASE=str(project.parent / "pi05_base")
    )

    assert result.returncode == 0, result.stderr
    stats = json.loads((meta / "stats.json").read_text())
    for key in ("action", "observation.state"):
        assert stats[key]["q01"][0] == pytest.approx(0.99) and stats[key]["q99"][5] == pytest.approx(103.01)
        assert stats[key]["mean"] == [0.0] * 6  # the other stats are kept
    assert (meta / "stats.json.orig").read_text() == original

    # A dataset that has them is left alone.
    (meta / "stats.json").write_text(json.dumps(stats))
    before = (meta / "stats.json").stat().st_mtime_ns
    shutil.rmtree(project / "outputs")
    assert (
        _run(
            project,
            "train_pi05_lora.sbatch",
            DATASET="pick_cube",
            PI05_BASE=str(project.parent / "pi05_base"),
        ).returncode
        == 0
    )
    assert (meta / "stats.json").stat().st_mtime_ns == before


def test_a_second_submit_resumes(project: Path) -> None:
    assert _run(project, "train_act.sbatch", DATASET="pick_cube").returncode == 0
    assert _run(project, "train_act.sbatch", DATASET="pick_cube").returncode == 0
    argv = json.loads((project / "argv.json").read_text())
    assert "--resume=true" in argv
    assert argv[0].startswith("--config_path=") and argv[0].endswith("train_config.json")


def test_settings_come_from_env_and_the_command_line_wins(project: Path) -> None:
    env_file = project / ".env"
    env_file.write_text(
        env_file.read_text().replace("DATASET=\n", "DATASET=nope\n").replace("ACT_BATCH=32", "ACT_BATCH=7")
    )
    assert _run(project, "train_act.sbatch").returncode == 1  # .env's dataset doesn't exist

    result = _run(project, "train_act.sbatch", DATASET="pick_cube")
    assert result.returncode == 0, result.stderr
    argv = json.loads((project / "argv.json").read_text())
    assert "--batch_size=7" in argv
    assert f"--dataset.root={project.parent}/data/pick_cube" in argv


def test_missing_dataset_and_lora_without_a_checkpoint_explain_themselves(project: Path) -> None:
    result = _run(project, "train_act.sbatch")
    assert result.returncode == 1 and "set DATASET" in result.stderr
    result = _run(project, "train_act_lora.sbatch", DATASET="pick_cube")
    assert result.returncode == 1 and "ACT_INIT" in result.stderr


def test_serve_runs_the_policy_server_with_the_policy_from_env(project: Path) -> None:
    (project / "lelab_policy_server.py").write_text(
        "import json, os, sys\nopen(os.environ['ARGV_OUT'], 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    result = _run(project, "serve_policy.sbatch")
    assert result.returncode == 1 and "no trained policy" in result.stderr

    assert _run(project, "train_act.sbatch", DATASET="pick_cube").returncode == 0
    result = _run(project, "serve_policy.sbatch")  # POLICY empty: the newest run
    assert result.returncode == 0, result.stderr
    argv = json.loads((project / "argv.json").read_text())
    assert argv[:2] == ["--policy", f"{project}/policies/act_full_pick_cube"]
    assert "--port" in argv and "5555" in argv

    result = _run(project, "serve_policy.sbatch", POLICY="policies/act_full_pick_cube", SERVE_PORT="6000")
    assert result.returncode == 0, result.stderr
    argv = json.loads((project / "argv.json").read_text())
    assert argv[1] == f"{project}/policies/act_full_pick_cube" and "6000" in argv

    # A training run folder serves its last checkpoint; a Hub id is passed through.
    result = _run(project, "serve_policy.sbatch", POLICY="lerobot/smolvla_base")
    assert result.returncode == 0, result.stderr
    assert json.loads((project / "argv.json").read_text())[1] == "lerobot/smolvla_base"


def test_without_lelab_the_scripts_use_the_submit_folder(project: Path, tmp_path: Path) -> None:
    """Unrendered scripts (copied by hand) fall back to the folder sbatch ran in."""
    for name in remote_upload.TRAINING_KIT_SCRIPTS:
        shutil.copy(remote_upload.TRAINING_KIT_DIR / name, project / "slurm" / name)
    result = _run(project, "train_act.sbatch", cwd=project, DATASET=str(tmp_path / "data" / "pick_cube"))
    assert result.returncode == 0, result.stderr


def test_trackio_settings(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run(project, "train_act.sbatch", DATASET="pick_cube", TRACKIO_SPACE_ID="me/lelab-trackio")
    assert result.returncode == 0, result.stderr
    argv = json.loads((project / "argv.json").read_text())
    shutil.rmtree(project / "outputs")
    monkeypatch.setattr(sys, "argv", ["lerobot-train", *argv])
    cfg = _parse_training()
    assert cfg.trackio.space_id == "me/lelab-trackio" and cfg.trackio.private is True

    assert _run(project, "train_act.sbatch", DATASET="pick_cube", TRACKIO="0").returncode == 0
    assert "--trackio.enable=false" in json.loads((project / "argv.json").read_text())

    # A resumed run keeps tracking.
    assert _run(project, "train_act.sbatch", DATASET="pick_cube").returncode == 0
    argv = json.loads((project / "argv.json").read_text())
    assert "--resume=true" in argv and "--trackio.enable=true" in argv


def test_policy_copy_keeps_the_model_after_its_run_is_deleted(project: Path) -> None:
    published = project / "policies" / "act_full_pick_cube"
    (project / "lelab_policy_server.py").write_text("import sys\n")

    assert _run(project, "train_act.sbatch", DATASET="pick_cube").returncode == 0
    shutil.rmtree(project / "outputs")
    result = _run(project, "serve_policy.sbatch", POLICY="policies/act_full_pick_cube")
    assert result.returncode == 1 and "no longer exists" in result.stderr

    assert _run(project, "train_act.sbatch", DATASET="pick_cube", POLICY_COPY="1").returncode == 0
    assert not published.is_symlink()
    shutil.rmtree(project / "outputs")
    assert (published / "config.json").is_file()
    assert _run(project, "serve_policy.sbatch", POLICY="policies/act_full_pick_cube").returncode == 0

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

import lerobot.scripts.lerobot_train  # noqa: F401  registers the policy configs, as the CLI does
from lelab import remote_upload
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
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
    return proj


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
        ACT_INIT=str(tmp / "act_init") if lora else "",
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    argv = json.loads((project / "argv.json").read_text())
    run = f"{policy_type}_{'lora' if lora else 'full'}_pick_cube"
    assert f"--output_dir={project}/outputs/{run}" in argv
    assert f"--dataset.root={tmp}/data/pick_cube" in argv
    # The finished policy is published for serve_policy.sbatch.
    assert (project / "policies" / run / "config.json").is_file()

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

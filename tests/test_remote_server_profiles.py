"""Tests for saved remote server profiles: persistence in lelab.utils.config
and the /remote-servers routes, including the origin guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROFILE = {"host": "10.2.100.14", "port": 22, "username": "student", "auth": "password"}


def test_profile_round_trips(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    saved = cfg.save_remote_server("cluster", {**PROFILE, "remote_dataset_dir": " projects/data "})

    assert saved["remote_dataset_dir"] == "projects/data"
    assert cfg.get_remote_server("cluster") == saved
    assert [s["name"] for s in cfg.list_remote_servers()] == ["cluster"]


def test_password_is_never_written_to_disk(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    cfg.save_remote_server("cluster", {**PROFILE, "password": "hunter2"})

    on_disk = (tmp_lerobot_home / "remote_servers" / "cluster.json").read_text()
    assert "hunter2" not in on_disk
    assert "password" not in json.loads(on_disk)


def test_update_keeps_fields_not_sent(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    cfg.save_remote_server("cluster", {**PROFILE, "remote_dataset_dir": "data"})
    updated = cfg.save_remote_server("cluster", {"port": "2222"})

    assert updated["port"] == 2222
    assert updated["host"] == "10.2.100.14"
    assert updated["remote_dataset_dir"] == "data"


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("../escape", PROFILE),
        ("cluster", {**PROFILE, "host": ""}),
        ("cluster", {**PROFILE, "host": "-oProxyCommand=x"}),
        ("cluster", {**PROFILE, "username": "two words"}),
        ("cluster", {**PROFILE, "port": 0}),
        ("cluster", {**PROFILE, "port": 70000}),
        ("cluster", {**PROFILE, "auth": "kerberos"}),
        ("cluster", {**PROFILE, "auth": "key", "key_path": ""}),
    ],
)
def test_invalid_profiles_are_rejected(tmp_lerobot_home: Path, name: str, data: dict) -> None:
    from lelab.utils import config as cfg

    with pytest.raises(ValueError):
        cfg.save_remote_server(name, data)
    assert cfg.list_remote_servers() == []


def test_delete_profile(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    cfg.save_remote_server("cluster", PROFILE)

    assert cfg.delete_remote_server("cluster") is True
    assert cfg.delete_remote_server("cluster") is False
    assert cfg.get_remote_server("cluster") is None


def test_routes_save_list_and_delete(tmp_lerobot_home: Path, client: TestClient) -> None:
    put = client.put("/remote-servers/cluster", json={**PROFILE, "password": "hunter2"})
    assert put.status_code == 200
    assert "password" not in put.json()

    listed = client.get("/remote-servers").json()["servers"]
    assert [s["name"] for s in listed] == ["cluster"]

    assert client.delete("/remote-servers/cluster").status_code == 200
    assert client.delete("/remote-servers/cluster").status_code == 404


def test_route_reports_an_invalid_profile(tmp_lerobot_home: Path, client: TestClient) -> None:
    response = client.put("/remote-servers/cluster", json={**PROFILE, "port": 0})
    assert response.status_code == 400
    assert "port" in response.json()["detail"]


@pytest.mark.parametrize(
    "origin",
    ["http://localhost:8080", "http://127.0.0.1:8000", "https://lerobot-lelab.hf.space", "http://testserver"],
)
def test_trusted_origins_pass(tmp_lerobot_home: Path, client: TestClient, origin: str) -> None:
    assert client.get("/remote-servers", headers={"origin": origin}).status_code == 200


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "https://lerobot-lelab.hf.space.evil.io"])
def test_other_origins_are_refused(tmp_lerobot_home: Path, client: TestClient, origin: str) -> None:
    response = client.put("/remote-servers/cluster", json=PROFILE, headers={"origin": origin})
    assert response.status_code == 403
    assert not (tmp_lerobot_home / "remote_servers" / "cluster.json").exists()

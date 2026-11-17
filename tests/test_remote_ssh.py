"""Tests for lelab.remote_ssh: authentication options, host-key confirmation,
session bookkeeping and the connect routes. paramiko's SSHClient is replaced by
a fake, so nothing touches the network."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import paramiko
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

PROFILE = {"host": "10.2.100.14", "port": 22, "username": "student", "auth": "password"}
SERVER_KEY = paramiko.ECDSAKey.generate()


class FakeTransport:
    def __init__(self) -> None:
        self.active = True
        self.keepalive = None

    def is_active(self) -> bool:
        return self.active

    def set_keepalive(self, interval: int) -> None:
        self.keepalive = interval


class FakeSftp:
    def normalize(self, path: str) -> str:
        return "/raid/home/student"

    def close(self) -> None:
        pass


class FakeClient:
    """Stand-in for paramiko.SSHClient. `mode` picks what connect() does."""

    mode = "ok"
    known = False
    instances: list[FakeClient] = []

    def __init__(self) -> None:
        self.policy: paramiko.MissingHostKeyPolicy | None = None
        self.host_keys = paramiko.HostKeys()
        self.connect_kwargs: dict[str, Any] = {}
        self.transport = FakeTransport()
        self.closed = False
        FakeClient.instances.append(self)

    def load_system_host_keys(self) -> None:
        pass

    def load_host_keys(self, path: str) -> None:
        FakeClient.known = True

    def set_missing_host_key_policy(self, policy: paramiko.MissingHostKeyPolicy) -> None:
        self.policy = policy

    def get_host_keys(self) -> paramiko.HostKeys:
        return self.host_keys

    def save_host_keys(self, path: str) -> None:
        self.host_keys.save(path)

    def connect(self, **kwargs: Any) -> None:
        self.connect_kwargs = kwargs
        logging.getLogger("paramiko.transport").debug("auth with %s", kwargs)
        if not FakeClient.known:
            assert self.policy is not None
            self.policy.missing_host_key(self, kwargs["hostname"], SERVER_KEY)
        if FakeClient.mode == "auth_fail":
            raise paramiko.AuthenticationException("bad password")
        if FakeClient.mode == "timeout":
            raise TimeoutError()

    def get_transport(self) -> FakeTransport:
        return self.transport

    def open_sftp(self) -> FakeSftp:
        return FakeSftp()

    def close(self) -> None:
        self.closed = True
        self.transport.active = False


@pytest.fixture(autouse=True)
def _fake_ssh(tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lelab import remote_ssh
    from lelab.utils import config as cfg

    FakeClient.mode = "ok"
    FakeClient.known = True
    FakeClient.instances = []
    monkeypatch.setattr(remote_ssh.paramiko, "SSHClient", FakeClient)
    monkeypatch.setattr(remote_ssh, "_sessions", {})
    cfg.save_remote_server("cluster", PROFILE)


def test_password_login_sends_only_the_password() -> None:
    from lelab import remote_ssh

    result = remote_ssh.connect("cluster", SecretStr("hunter2"))

    kwargs = FakeClient.instances[-1].connect_kwargs
    assert kwargs["password"] == "hunter2"
    assert kwargs["look_for_keys"] is False
    assert kwargs["allow_agent"] is False
    assert result["connected"] is True
    assert result["home"] == "/raid/home/student"


def test_password_login_without_a_password_asks_for_one() -> None:
    from lelab import remote_ssh

    with pytest.raises(remote_ssh.RemoteServerError) as exc:
        remote_ssh.connect("cluster", None)
    assert exc.value.code == "password_required"


def test_key_login_uses_the_key_file(tmp_lerobot_home: Path) -> None:
    from lelab import remote_ssh
    from lelab.utils import config as cfg

    cfg.save_remote_server("cluster", {"auth": "key", "key_path": "~/.ssh/id_ed25519"})
    remote_ssh.connect("cluster", None)

    kwargs = FakeClient.instances[-1].connect_kwargs
    assert kwargs["key_filename"].endswith(".ssh/id_ed25519")
    assert "~" not in kwargs["key_filename"]
    assert "password" not in kwargs


def test_unknown_host_key_needs_confirmation(tmp_lerobot_home: Path) -> None:
    from lelab import remote_ssh

    FakeClient.known = False
    with pytest.raises(remote_ssh.RemoteServerError) as exc:
        remote_ssh.connect("cluster", SecretStr("pw"))
    assert exc.value.status_code == 409
    assert exc.value.code == "unknown_host_key"
    expected = remote_ssh.fingerprint(SERVER_KEY)
    assert exc.value.extra["fingerprint"] == expected
    assert FakeClient.instances[-1].closed

    with pytest.raises(remote_ssh.RemoteServerError):
        remote_ssh.connect("cluster", SecretStr("pw"), trust_fingerprint="SHA256:somethingelse")

    assert remote_ssh.connect("cluster", SecretStr("pw"), trust_fingerprint=expected)["connected"]
    known_hosts = (tmp_lerobot_home / "remote_known_hosts").read_text()
    assert "10.2.100.14" in known_hosts


def test_failed_login_is_a_401() -> None:
    from lelab import remote_ssh

    FakeClient.mode = "auth_fail"
    with pytest.raises(remote_ssh.RemoteServerError) as exc:
        remote_ssh.connect("cluster", SecretStr("wrong"))
    assert exc.value.status_code == 401
    assert remote_ssh.status("cluster") == {"connected": False}


def test_password_never_appears_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    from lelab import remote_ssh

    caplog.set_level(logging.DEBUG, logger="lelab")
    remote_ssh.connect("cluster", SecretStr("hunter2"))
    remote_ssh.disconnect("cluster")

    lelab_records = [r.getMessage() for r in caplog.records if r.name.startswith("lelab")]
    assert lelab_records
    assert not any("hunter2" in message for message in lelab_records)


def test_disconnect_is_refused_while_the_session_is_in_use() -> None:
    from lelab import remote_ssh

    remote_ssh.connect("cluster", SecretStr("pw"))
    with remote_ssh.lease("cluster"):
        with pytest.raises(remote_ssh.RemoteServerError) as exc:
            remote_ssh.disconnect("cluster")
        assert exc.value.code == "in_use"
    assert remote_ssh.disconnect("cluster") == {"connected": False}


def test_a_dropped_session_reconnects_with_the_remembered_password() -> None:
    from lelab import remote_ssh

    remote_ssh.connect("cluster", SecretStr("pw"))
    FakeClient.instances[-1].transport.active = False

    session = remote_ssh.ensure_connected("cluster")

    assert len(FakeClient.instances) == 2
    assert FakeClient.instances[-1].connect_kwargs["password"] == "pw"
    assert session.client is FakeClient.instances[-1]


def test_ensure_connected_without_a_session() -> None:
    from lelab import remote_ssh

    with pytest.raises(remote_ssh.RemoteServerError) as exc:
        remote_ssh.ensure_connected("cluster")
    assert exc.value.code == "not_connected"


def test_an_agent_login_reconnects_on_its_own_after_a_restart() -> None:
    """No session (LeLab restarted): a login that needs no secret just reconnects."""
    from lelab import remote_ssh
    from lelab.utils import config as cfg

    cfg.save_remote_server("cluster", {**PROFILE, "auth": "agent"})

    session = remote_ssh.ensure_connected("cluster")

    assert session.client is FakeClient.instances[-1]
    assert remote_ssh.status("cluster")["connected"] is True


def test_a_failed_quiet_reconnect_is_not_connected() -> None:
    from lelab import remote_ssh
    from lelab.utils import config as cfg

    cfg.save_remote_server("cluster", {**PROFILE, "auth": "agent"})
    FakeClient.mode = "auth_fail"

    with pytest.raises(remote_ssh.RemoteServerError) as exc:
        remote_ssh.ensure_connected("cluster")
    assert exc.value.code == "not_connected"
    assert exc.value.status_code == 409


def test_connect_route_reports_the_fingerprint(client: TestClient) -> None:
    FakeClient.known = False

    response = client.post("/remote-servers/cluster/connect", json={"password": "pw"})

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "unknown_host_key"
    assert detail["fingerprint"].startswith("SHA256:")


def test_connect_and_list_routes(client: TestClient) -> None:
    assert client.post("/remote-servers/cluster/connect", json={"password": "pw"}).status_code == 200

    servers = client.get("/remote-servers").json()["servers"]
    assert servers[0]["session"]["connected"] is True
    assert client.get("/remote-servers/cluster/status").json()["connected"] is True
    assert client.post("/remote-servers/cluster/disconnect").json() == {"connected": False}


def test_routes_for_an_unknown_server_are_404(client: TestClient) -> None:
    assert client.post("/remote-servers/nope/connect", json={"password": "pw"}).status_code == 404
    assert client.get("/remote-servers/nope/status").status_code == 404

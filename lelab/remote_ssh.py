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

"""SSH sessions to the user's own servers (a GPU box or a Slurm login node).

One paramiko session per saved profile (lelab/utils/config.py), opened from the
UI and kept in memory for as long as LeLab runs. The session carries both
features built on it: SFTP dataset uploads (lelab/remote_upload.py) and the
"direct-tcpip" channel to a policy server on the server's network
(lelab/remote_policy.py), which is what `ssh -N -L port:node:port user@login`
does, without opening a local port.

Passwords live only in this process's memory, so a dropped connection can be
reopened without asking again; they are never written to disk or logged.
Unknown host keys are not accepted silently: the first connect fails with the
key's fingerprint, the UI shows it, and a repeat request carrying that
fingerprint saves the key to LeLab's own known_hosts file.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import paramiko
from pydantic import BaseModel, SecretStr

from .utils import config

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_S = 10
KEEPALIVE_S = 15


class RemoteServerError(Exception):
    """A failure to report to the UI, with the HTTP status the route should use."""

    def __init__(self, message: str, status_code: int = 400, code: str | None = None, **extra: Any):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.extra = extra

    def detail(self) -> dict[str, Any] | str:
        if self.code is None and not self.extra:
            return self.message
        return {"message": self.message, "code": self.code, **self.extra}


class ConnectRequest(BaseModel):
    # Password, or the key's passphrase with auth="key". SecretStr keeps it out of reprs and logs.
    password: SecretStr | None = None
    # SHA256 fingerprint the user confirmed after a first connect reported an unknown host key.
    trust_fingerprint: str | None = None


@dataclass
class _Session:
    name: str
    profile: dict[str, Any]
    client: paramiko.SSHClient
    password: SecretStr | None
    connected_at: float
    home: str | None = None
    # Features currently streaming over this session (uploads, inference runs).
    users: int = 0


_sessions: dict[str, _Session] = {}
_lock = threading.Lock()


def fingerprint(key: paramiko.PKey) -> str:
    """OpenSSH-style SHA256 fingerprint, as `ssh` prints it."""
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode().rstrip("=")


class _UnknownHostKeyError(Exception):
    def __init__(self, hostname: str, key: paramiko.PKey):
        super().__init__(hostname)
        self.hostname = hostname
        self.key = key


class _ConfirmHostKeyPolicy(paramiko.MissingHostKeyPolicy):
    """Accept an unknown host key only when its fingerprint is the one the user confirmed."""

    def __init__(self, trusted_fingerprint: str | None):
        self.trusted = (trusted_fingerprint or "").strip()

    def missing_host_key(self, client: paramiko.SSHClient, hostname: str, key: paramiko.PKey) -> None:
        if self.trusted and self.trusted == fingerprint(key):
            client.get_host_keys().add(hostname, key.get_name(), key)
            path = config.REMOTE_KNOWN_HOSTS_PATH
            os.makedirs(os.path.dirname(path), exist_ok=True)
            client.save_host_keys(path)
            logger.info("Trusted the host key of %s (%s)", hostname, fingerprint(key))
            return
        raise _UnknownHostKeyError(hostname, key)


def _profile(name: str) -> dict[str, Any]:
    profile = config.get_remote_server(name)
    if profile is None:
        raise RemoteServerError(f"No saved server named {name!r}", 404)
    return profile


def _new_client(trust_fingerprint: str | None) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    # Keys the user already accepted with their own ssh client count as known.
    try:
        client.load_system_host_keys()
    except (OSError, paramiko.SSHException) as exc:
        logger.debug("Could not read ~/.ssh/known_hosts: %s", exc)
    if os.path.exists(config.REMOTE_KNOWN_HOSTS_PATH):
        client.load_host_keys(config.REMOTE_KNOWN_HOSTS_PATH)
    client.set_missing_host_key_policy(_ConfirmHostKeyPolicy(trust_fingerprint))
    return client


def _auth_kwargs(profile: dict[str, Any], password: SecretStr | None) -> dict[str, Any]:
    secret = password.get_secret_value() if password is not None else None
    auth = profile["auth"]
    if auth == "password":
        if not secret:
            raise RemoteServerError("Enter the password for this server.", 400, code="password_required")
        # Only the password: trying every local key first can use up the
        # server's MaxAuthTries before the password is even sent. paramiko falls
        # back to keyboard-interactive when the server asks for that instead.
        return {"password": secret, "look_for_keys": False, "allow_agent": False}
    if auth == "key":
        return {
            "key_filename": os.path.expanduser(profile["key_path"]),
            "passphrase": secret,
            "look_for_keys": False,
            "allow_agent": False,
        }
    return {"allow_agent": True, "look_for_keys": True}


def _open_client(
    profile: dict[str, Any], password: SecretStr | None, trust: str | None
) -> paramiko.SSHClient:
    host, port = profile["host"], profile["port"]
    client = _new_client(trust)
    try:
        client.connect(
            hostname=host,
            port=port,
            username=profile["username"],
            timeout=CONNECT_TIMEOUT_S,
            banner_timeout=CONNECT_TIMEOUT_S,
            auth_timeout=2 * CONNECT_TIMEOUT_S,
            **_auth_kwargs(profile, password),
        )
    except _UnknownHostKeyError as exc:
        client.close()
        raise RemoteServerError(
            f"{host} is not a known host yet. Check that this fingerprint is your server's, then connect again "
            "to trust it.",
            409,
            code="unknown_host_key",
            fingerprint=fingerprint(exc.key),
            key_type=exc.key.get_name(),
        ) from None
    except paramiko.BadHostKeyException as exc:
        client.close()
        raise RemoteServerError(
            f"The host key of {host} changed since you trusted it ({fingerprint(exc.key)}). That happens when "
            "the server is reinstalled, but it can also mean someone is in between. Remove the old entry "
            f"from {config.REMOTE_KNOWN_HOSTS_PATH} or ~/.ssh/known_hosts only if you know why it changed.",
            409,
            code="host_key_changed",
        ) from None
    except paramiko.AuthenticationException:
        client.close()
        hint = (
            "Wrong user name or password." if profile["auth"] == "password" else "The server refused the key."
        )
        raise RemoteServerError(f"Login to {profile['username']}@{host} failed. {hint}", 401) from None
    except TimeoutError:
        client.close()
        raise RemoteServerError(f"{host}:{port} did not answer within {CONNECT_TIMEOUT_S} s.", 504) from None
    except (OSError, paramiko.SSHException) as exc:
        client.close()
        raise RemoteServerError(f"Could not reach {host}:{port}: {exc}", 502) from None

    transport = client.get_transport()
    if transport is not None:
        transport.set_keepalive(KEEPALIVE_S)
    return client


def _home_dir(client: paramiko.SSHClient) -> str | None:
    """The login's home directory, as SFTP sees it (None if SFTP is refused)."""
    try:
        sftp = client.open_sftp()
    except (OSError, paramiko.SSHException) as exc:
        logger.warning("SFTP is not available on this server: %s", exc)
        return None
    try:
        return sftp.normalize(".")
    except (OSError, paramiko.SSHException):
        return None
    finally:
        sftp.close()


def _is_active(session: _Session) -> bool:
    transport = session.client.get_transport()
    return transport is not None and transport.is_active()


def _describe(session: _Session | None) -> dict[str, Any]:
    if session is None or not _is_active(session):
        return {"connected": False}
    return {
        "connected": True,
        "connected_at": session.connected_at,
        "home": session.home,
        "in_use": session.users > 0,
    }


def connect(
    name: str, password: SecretStr | None = None, trust_fingerprint: str | None = None
) -> dict[str, Any]:
    """Open (or reopen) the session for a saved server and keep it."""
    profile = _profile(name)
    with _lock:
        old = _sessions.get(name)
    if password is None and old is not None and old.profile == profile:
        password = old.password  # reconnect after a drop without asking again

    client = _open_client(profile, password, trust_fingerprint)
    session = _Session(name, profile, client, password, time.time(), home=_home_dir(client))
    with _lock:
        old = _sessions.get(name)
        if old is not None and old.users > 0 and _is_active(old):
            # Something streams over the old session: keep it and drop the new one.
            client.close()
            return status(name)
        _sessions[name] = session
    if old is not None:
        old.client.close()
    logger.info("Connected to %s (%s@%s:%s)", name, profile["username"], profile["host"], profile["port"])
    return status(name)


def ensure_connected(name: str) -> _Session:
    """The live session for `name`, reconnecting with the remembered password if it dropped."""
    with _lock:
        session = _sessions.get(name)
    if session is not None and _is_active(session):
        return session
    if session is None:
        raise RemoteServerError(
            f"Not connected to {name}. Connect to the server first.", 409, code="not_connected"
        )
    profile = _profile(name)
    if profile["auth"] == "password" and session.password is None:
        raise RemoteServerError(
            f"The connection to {name} dropped. Connect again.", 409, code="not_connected"
        )
    connect(name, session.password)
    with _lock:
        return _sessions[name]


def disconnect(name: str) -> dict[str, Any]:
    with _lock:
        session = _sessions.get(name)
        if session is not None and session.users > 0 and _is_active(session):
            raise RemoteServerError(
                f"{name} is in use by an upload or an inference run. Stop that first.", 409, code="in_use"
            )
        _sessions.pop(name, None)
    if session is not None:
        session.client.close()
        logger.info("Disconnected from %s", name)
    return {"connected": False}


def status(name: str) -> dict[str, Any]:
    _profile(name)
    with _lock:
        session = _sessions.get(name)
    return _describe(session)


def statuses() -> dict[str, dict[str, Any]]:
    with _lock:
        sessions = dict(_sessions)
    return {name: _describe(session) for name, session in sessions.items()}


def close_all() -> None:
    """Close every session (server shutdown)."""
    with _lock:
        sessions = list(_sessions.values())
        _sessions.clear()
    for session in sessions:
        try:
            session.client.close()
        except Exception as exc:
            logger.warning("Error closing the session to %s: %s", session.name, exc)


class _Lease:
    """Marks a session in use while an upload or a run streams over it, so it can't be disconnected."""

    def __init__(self, session: _Session):
        self.session = session

    def __enter__(self) -> _Session:
        with _lock:
            self.session.users += 1
        return self.session

    def __exit__(self, *exc: object) -> None:
        with _lock:
            self.session.users = max(0, self.session.users - 1)


def lease(name: str) -> _Lease:
    return _Lease(ensure_connected(name))


def open_sftp(session: _Session) -> paramiko.SFTPClient:
    try:
        return session.client.open_sftp()
    except (OSError, paramiko.SSHException) as exc:
        raise RemoteServerError(
            f"{session.name} refused SFTP ({exc}). Ask the administrator how to copy files to it.", 502
        ) from None


def open_policy_channel(session: _Session, host: str, port: int, timeout: float = 10.0) -> paramiko.Channel:
    """A socket-like channel from the server to host:port on its network (an SSH tunnel)."""
    transport = session.client.get_transport()
    if transport is None or not transport.is_active():
        raise RemoteServerError(f"The connection to {session.name} dropped. Connect again.", 409)
    try:
        return transport.open_channel("direct-tcpip", (host, port), ("127.0.0.1", 0), timeout=timeout)
    except paramiko.ChannelException as exc:
        if exc.code == paramiko.common.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED:
            raise RemoteServerError(
                f"{session.name} does not allow SSH tunnels. If you are on the same network as the GPU node, "
                "connect directly to it instead.",
                502,
            ) from None
        raise RemoteServerError(
            f"Nothing answers at {host}:{port} as seen from {session.name}. Is the policy server running, and "
            "are the node IP and port the ones its log printed?",
            502,
        ) from None
    except (OSError, paramiko.SSHException) as exc:
        raise RemoteServerError(f"Could not open a tunnel to {host}:{port}: {exc}", 502) from None

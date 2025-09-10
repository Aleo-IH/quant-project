import types
from pathlib import Path
import socket

import pytest

from models.RL.train import TensorBoardServer


def test_tensorboard_server_start_stop(monkeypatch, tmp_path: Path):
    # Fake Popen that simulates a long-running process
    class FakeProc:
        def __init__(self):
            self._running = True

        def poll(self):
            return None if self._running else 0

        def terminate(self):
            self._running = False

        def kill(self):
            self._running = False

        def wait(self, timeout=None):
            self._running = False
            return 0

    def fake_popen(*args, **kwargs):
        return FakeProc()

    # Simulate port not accepting connections
    def fake_create_connection(*args, **kwargs):
        raise OSError("port not ready")

    import subprocess

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(socket, "create_connection", fake_create_connection)

    tbs = TensorBoardServer(logdir=str(tmp_path))
    assert tbs.start() is True  # returns True even if port isn't ready yet
    assert tbs.stop() is True

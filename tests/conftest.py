import socket

import pytest


@pytest.fixture(autouse=True)
def no_test_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not use network access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)

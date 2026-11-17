import socket

import pytest
import requests

BLOCKED = "Offline test attempted a network connection"


def test_raw_socket_connections_are_blocked():
    with pytest.raises(RuntimeError, match=BLOCKED):
        socket.create_connection(("generativelanguage.googleapis.com", 443), timeout=1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match=BLOCKED):
            sock.connect(("api.openalex.org", 443))
    finally:
        sock.close()


def test_http_clients_cannot_reach_the_network():
    # requests wraps the guard error; it must never produce a real HTTP response.
    with pytest.raises(Exception, match=BLOCKED):
        requests.get("https://api.openalex.org/works", timeout=1)

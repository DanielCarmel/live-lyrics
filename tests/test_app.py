import shutil
import ssl

import pytest

from live_lyrics.app import ensure_cert


@pytest.mark.skipif(not shutil.which("openssl"), reason="openssl not installed")
def test_ensure_cert_makes_a_loadable_pair(tmp_path):
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    assert ensure_cert(cert, key)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)   # raises if the pair is broken
    assert ensure_cert(cert, key)    # second call: already there, nothing rewritten

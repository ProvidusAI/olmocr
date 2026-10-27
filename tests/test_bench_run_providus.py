"""Tests for the providus-vl-mini runner. A fake /predict server; no model and no benchmark data."""

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image
from pypdf import PdfReader, PdfWriter

from olmocr.bench.runners.run_providus import run_providus


@pytest.fixture
def pdf_path(tmp_path):
    """A 2-page PDF with no text. Page 1 is 200 x 200 pt. Page 2 is 300 x 100 pt."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=300, height=100)
    path = tmp_path / "two_pages.pdf"
    with open(path, "wb") as f:
        writer.write(f)
    return str(path)


@pytest.fixture
def fake_server():
    """fake_server(answers) starts a /predict server. It gives the answers in order and repeats the last one."""
    servers = []

    def start(answers):
        bodies = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                status, answer = answers[min(len(bodies), len(answers)) - 1]
                raw = json.dumps(answer).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_port}", bodies

    yield start
    for server in servers:
        server.shutdown()


def image_size(body):
    return Image.open(io.BytesIO(base64.b64decode(body["image_base64"]))).size


def test_returns_the_markdown_and_sends_png_and_one_page_pdf(pdf_path, fake_server):
    url, bodies = fake_server([(200, {"status": "success", "markdown": "# Title\n\ntext"})])
    assert run_providus(pdf_path, server_url=url) == "# Title\n\ntext"
    assert image_size(bodies[0]) == (400, 400)  # 200 pt at 144 dpi
    assert len(PdfReader(io.BytesIO(base64.b64decode(bodies[0]["pdf_base64"]))).pages) == 1


def test_page_num_selects_the_page(pdf_path, fake_server):
    url, bodies = fake_server([(200, {"status": "success", "markdown": "page 2"})])
    run_providus(pdf_path, page_num=2, server_url=url)
    assert image_size(bodies[0]) == (600, 200)


def test_the_server_address_can_come_from_the_environment(pdf_path, fake_server, monkeypatch):
    url, _ = fake_server([(200, {"status": "success", "markdown": "ok"})])
    monkeypatch.setenv("PROVIDUS_PARSE_URL", url)
    assert run_providus(pdf_path) == "ok"


def test_an_empty_answer_is_returned_as_it_is(pdf_path, fake_server):
    url, _ = fake_server([(200, {"status": "success", "markdown": ""})])
    assert run_providus(pdf_path, server_url=url) == ""


def test_a_failed_request_is_tried_again(pdf_path, fake_server):
    url, bodies = fake_server([(500, {}), (200, {"status": "success", "markdown": "ok"})])
    assert run_providus(pdf_path, server_url=url, retry_wait=0) == "ok"
    assert len(bodies) == 2


def test_raises_after_three_failures_and_writes_nothing(pdf_path, fake_server):
    url, bodies = fake_server([(500, {}), (200, {"status": "error", "error": "boom"})])
    with pytest.raises(RuntimeError, match="failed 3 times"):
        run_providus(pdf_path, server_url=url, retry_wait=0)
    assert len(bodies) == 3


def test_a_page_outside_the_pdf_raises(pdf_path):
    with pytest.raises(ValueError, match="page 3"):
        run_providus(pdf_path, page_num=3, server_url="http://127.0.0.1:9")

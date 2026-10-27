"""providus-vl-mini runner for olmOCR-Bench.

The runner posts each page to a providus_parse /predict server (pip install providus-parse-tools). That server
reads the page with providus-vl-mini through llama-server. The "providus-vl-mini" section of olmocr/bench/README.md
shows how to start both servers.

convert splits method options on ":", so a URL cannot be a method option. Set PROVIDUS_PARSE_URL instead
(default http://127.0.0.1:8100).
Run convert with --failfast: without it, convert writes an empty file for a page that failed.
"""

import base64
import io
import os
import time

import httpx
import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter


def _page_body(pdf_path: str, page_num: int, dpi: int) -> dict:
    """The /predict body: the page as a PNG at `dpi`, and the same page as a one-page PDF (its text layer)."""
    doc = pdfium.PdfDocument(pdf_path)
    try:
        if not 1 <= page_num <= len(doc):
            raise ValueError(f"page {page_num} is not in {pdf_path} ({len(doc)} pages)")
        image = doc[page_num - 1].render(scale=dpi / 72).to_pil()
    finally:
        doc.close()
    png = io.BytesIO()
    image.save(png, format="PNG")
    writer = PdfWriter()
    writer.add_page(PdfReader(pdf_path).pages[page_num - 1])
    pdf = io.BytesIO()
    writer.write(pdf)
    return {
        "image_base64": base64.b64encode(png.getvalue()).decode(),
        "pdf_base64": base64.b64encode(pdf.getvalue()).decode(),
    }


def run_providus(
    pdf_path: str,
    page_num: int = 1,
    server_url: str | None = None,
    dpi: int = 144,
    timeout: float = 900.0,
    attempts: int = 3,
    retry_wait: float = 5.0,
) -> str:
    """Return the markdown of one page. Raise after `attempts` failed requests: with --failfast, convert then writes no file."""
    url = (server_url or os.environ.get("PROVIDUS_PARSE_URL") or "http://127.0.0.1:8100").rstrip("/")
    body = _page_body(pdf_path, page_num, dpi)
    error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = httpx.post(f"{url}/predict", json=body, timeout=timeout)
            response.raise_for_status()
            answer = response.json()
            if answer.get("status") == "success":
                return answer["markdown"]
            error = RuntimeError(answer.get("error") or f"status {answer.get('status')!r}")
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            error = exc
        if attempt < attempts:
            time.sleep(retry_wait * attempt)
    raise RuntimeError(f"/predict failed {attempts} times on page {page_num} of {pdf_path}: {error}") from error

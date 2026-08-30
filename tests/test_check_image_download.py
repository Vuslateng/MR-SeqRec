import ssl
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from scripts.check_image_download import (
    _magic_name,
    collect_urls,
    http_status,
    summarize,
    verdict,
)

_PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 60


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ok.png":
            body = _PNG_HEADER
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


# ---------------------------------------------------------------- URL 收集


def test_collect_urls_caps_and_stats():
    rows = [
        {"parent_asin": f"P{i}", "title": f"t{i}",
         "images": [{"large": f"http://h/{i}/a.jpg"}, {"large": f"http://h/{i}/b.jpg"}]}
        for i in range(5)
    ]
    # 3 物品 × 2 URL = 6，max_urls=5 -> 收全量洗牌后随机截断为 5（非按物品先到先得）
    urls, st = collect_urls(rows, sample=3, max_urls=5)
    assert st["n_items"] == 3
    assert st["pct_with_image"] == 100.0
    assert len(urls) == 5
    assert st["n_items_in_probe"] == 3  # 只截 1 个 URL，3 个来源物品都保留
    # sample 截断
    urls2, st2 = collect_urls(rows, sample=2, max_urls=100)
    assert len(urls2) == 4 and st2["n_items"] == 2
    assert st2["urls_per_item_mean"] == 2.0
    assert st2["n_items_in_probe"] == 2
    # 无图物品
    urls3, st3 = collect_urls([{"parent_asin": "X"}], sample=1, max_urls=100)
    assert urls3 == [] and st3["pct_with_image"] == 0.0
    assert st3["n_items_in_probe"] == 0


def test_collect_urls_reservoir_seed():
    rows = [
        {"parent_asin": f"P{i}", "images": [{"large": f"http://h/{i}/x.jpg"}]}
        for i in range(10)
    ]
    # 同种子重跑一致（蓄水池抽样的可复现性）
    u1, _ = collect_urls(rows, sample=3, max_urls=100, seed=42)
    u2, _ = collect_urls(rows, sample=3, max_urls=100, seed=42)
    assert u1 == u2
    assert len(u1) == 3  # 3 个物品各 1 URL
    # sample >= 总数时返回全部（蓄水池不丢尾部）
    u4, st4 = collect_urls(rows, sample=100, max_urls=100, seed=42)
    assert len(u4) == 10 and st4["n_items"] == 10


# ---------------------------------------------------------------- 魔数


def test_magic_name_formats():
    assert _magic_name(b"\xff\xd8\xff\xe0") == "jpeg"
    assert _magic_name(_PNG_HEADER) == "png"
    assert _magic_name(b"GIF89a") == "gif"
    assert _magic_name(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
    assert _magic_name(b"\x00\x00\x00\x1cftypavif") == "avif"
    assert _magic_name(b"<html>") is None
    assert _magic_name(b"") is None


# ---------------------------------------------------------------- HTTP 探测


def test_http_status_local(local_server):
    code, first, fail = http_status(f"{local_server}/ok.png", 3.0, 2048)
    assert code == 200
    assert _magic_name(first) == "png"
    assert fail is None
    # 404 = 服务器明确响应了 -> fail_kind "http"，不算网络层失败
    code2, first2, fail2 = http_status(f"{local_server}/missing.png", 3.0, 2048)
    assert code2 == 404
    assert first2 == b""
    assert fail2 == "http"


def test_http_status_conn_refused():
    # 127.0.0.1 上的空端口 -> 网络层失败（本机可能是超时而非拒绝，正确分类为 timeout）
    code, first, fail = http_status("http://127.0.0.1:1/x.jpg", 2.0, 2048)
    assert code is None
    assert fail in {"conn", "timeout", "other"}


def test_http_status_ssl_classified(monkeypatch):
    # SSL 错误须单独分类，不得混入 conn（防"证书问题"误判为网络屏蔽）
    def fake_urlopen(req, timeout):
        raise urllib.error.URLError(ssl.SSLError("certificate verify failed"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    code, first, fail = http_status("https://example.com/x.jpg", 3.0, 2048)
    assert code is None and first == b"" and fail == "ssl"


# ---------------------------------------------------------------- 汇总


def test_summarize_counts():
    urls = ["http://h1/a.jpg", "http://h1/b.jpg", "http://h2/c.png"]
    results = [
        ("http://h1/a.jpg", 200, _PNG_HEADER, None),
        ("http://h1/b.jpg", 404, b"", "http"),
        ("http://h2/c.png", None, b"", "conn"),
    ]
    s = summarize(urls, results)
    assert s["n_urls"] == 3 and s["n_ok"] == 1
    assert s["pct_ok"] == round(100.0 / 3, 2)
    assert s["pct_valid_image"] == 100.0 and s["n_magic_tried"] == 1
    assert s["by_status"] == {"200": 1, "404": 1}
    assert s["by_failure"] == {"conn": 1}
    assert s["by_host"]["h1"]["pct_ok"] == 50.0


def test_verdict_thresholds():
    assert "可建" in verdict(95.0)
    assert "中等" in verdict(60.0)
    assert "偏低" in verdict(20.0)

import asyncio
import ipaddress
import json
import os
import socket
import subprocess
from typing import Any
from urllib.parse import urlparse

import httpx
from fastmcp import FastMCP

mcp = FastMCP(
    "Business OSS Gateway",
    instructions=(
        "Open-source business toolbox for BidSathi, MarketBridge, OneKivo, Founder Features, "
        "news pages and AI Biopics. Prefer the narrowest tool for the task. Only public HTTP(S) "
        "targets are allowed; this server will not access private/internal network addresses."
    ),
)

MAX_TEXT = 120_000
USER_AGENT = "Business-OSS-Gateway/1.0"

STACK = {
    "activepieces": {"mode": "service_definition", "businesses": ["all"], "status": "boot_tested_separately"},
    "crawl4ai": {"mode": "manufact_direct", "businesses": ["BidSathi", "MarketBridge", "Founder Features"], "status": "ready"},
    "docling": {"mode": "separate_worker", "businesses": ["BidSathi"], "status": "separate_worker"},
    "searxng": {"mode": "public_instance_pool", "businesses": ["BidSathi", "news"], "status": "ready"},
    "espocrm": {"mode": "service_definition", "businesses": ["BidSathi", "MarketBridge", "Founder Features", "AI Biopics"], "status": "boot_tested_separately"},
    "baserow": {"mode": "service_definition", "businesses": ["all"], "status": "boot_tested_separately"},
    "playwright": {"mode": "manufact_direct", "businesses": ["BidSathi", "MarketBridge", "research"], "status": "ready"},
    "rsshub": {"mode": "public_instance", "businesses": ["news"], "status": "ready"},
    "ffmpeg": {"mode": "manufact_direct", "businesses": ["news", "AI Biopics"], "status": "ready"},
    "yt-dlp": {"mode": "manufact_direct_metadata", "businesses": ["news"], "status": "ready"},
    "whisper.cpp": {"mode": "separate_worker", "businesses": ["news", "AI Biopics"], "status": "separate_worker"},
    "comfyui": {"mode": "gpu_definition", "businesses": ["AI Biopics"], "status": "requires_gpu_for_generation"},
    "supabase": {"mode": "chatgpt_plugin", "businesses": ["OneKivo", "all"], "status": "connected"},
    "metabase": {"mode": "service_definition", "businesses": ["all"], "status": "boot_tested_separately"},
    "erpnext": {"mode": "service_definition", "businesses": ["MarketBridge"], "status": "boot_tested_separately"},
}


def _trim(value: str, limit: int = MAX_TEXT) -> str:
    value = value or ""
    return value if len(value) <= limit else value[:limit] + "\n...[truncated]"


def _public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("A public http:// or https:// URL is required.")
    host = parsed.hostname
    try:
        addresses = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Hostname could not be resolved: {host}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("Private, loopback, link-local, reserved and other non-public targets are blocked.")
    return url


@mcp.tool
def list_stack() -> dict[str, Any]:
    """List the 15 open-source tools and their current cloud execution modes."""
    return {"tools": STACK}


@mcp.tool
async def crawl4ai_extract(url: str) -> dict[str, Any]:
    """Extract a public webpage into clean LLM-friendly Markdown using Crawl4AI."""
    from crawl4ai import AsyncWebCrawler

    target = _public_url(url)
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=target)
    markdown = getattr(result, "markdown", "") or ""
    return {
        "ok": True,
        "url": target,
        "markdown": _trim(str(markdown)),
    }


@mcp.tool
async def playwright_extract(url: str, wait_ms: int = 1200) -> dict[str, Any]:
    """Open a public webpage in Chromium and return rendered title, final URL and visible body text."""
    from playwright.async_api import async_playwright

    target = _public_url(url)
    wait_ms = max(0, min(int(wait_ms), 10_000))
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(target, wait_until="domcontentloaded", timeout=60_000)
            if wait_ms:
                await page.wait_for_timeout(wait_ms)
            return {
                "ok": True,
                "url": page.url,
                "title": await page.title(),
                "text": _trim(await page.locator("body").inner_text()),
            }
        finally:
            await browser.close()


async def _searx_instances() -> list[str]:
    configured = [x.strip().rstrip("/") for x in os.getenv("SEARXNG_URLS", "").split(",") if x.strip()]
    if configured:
        return configured
    async with httpx.AsyncClient(timeout=12, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        response = await client.get("https://searx.space/data/instances.json")
        response.raise_for_status()
        data = response.json().get("instances", {})
    candidates: list[str] = []
    for base, meta in data.items():
        if not isinstance(base, str) or not base.startswith("https://"):
            continue
        if isinstance(meta, dict) and meta.get("network_type") not in {None, "normal"}:
            continue
        candidates.append(base.rstrip("/"))
        if len(candidates) >= 25:
            break
    return candidates


@mcp.tool
async def searxng_search(query: str, limit: int = 10) -> dict[str, Any]:
    """Search the web through a live SearXNG instance pool and return JSON results."""
    query = query.strip()
    if not query:
        raise ValueError("query is required")
    limit = max(1, min(int(limit), 20))
    errors: list[str] = []
    instances = await _searx_instances()
    async with httpx.AsyncClient(timeout=10, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        for base in instances:
            try:
                r = await client.get(base + "/search", params={"q": query, "format": "json"})
                if r.status_code != 200:
                    errors.append(f"{base}:{r.status_code}")
                    continue
                payload = r.json()
                rows = []
                for item in payload.get("results", [])[:limit]:
                    rows.append({
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "content": _trim(str(item.get("content") or ""), 1500),
                        "engine": item.get("engine"),
                        "publishedDate": item.get("publishedDate"),
                    })
                return {"ok": True, "instance": base, "query": query, "results": rows}
            except Exception as exc:
                errors.append(f"{base}:{type(exc).__name__}")
                if len(errors) >= 12:
                    break
    return {"ok": False, "query": query, "error": "No JSON-enabled SearXNG instance responded.", "attempts": errors}


@mcp.tool
async def rsshub_fetch(route: str) -> dict[str, Any]:
    """Fetch an RSSHub route from the public RSSHub service; pass a route such as /github/trending/daily."""
    route = route.strip()
    if not route.startswith("/"):
        route = "/" + route
    if ".." in route:
        raise ValueError("Invalid route")
    base = os.getenv("RSSHUB_URL", "https://rsshub.app").rstrip("/")
    async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        r = await client.get(base + route)
        r.raise_for_status()
    return {
        "ok": True,
        "route": route,
        "content_type": r.headers.get("content-type"),
        "body": _trim(r.text),
    }


@mcp.tool
async def ytdlp_metadata(url: str) -> dict[str, Any]:
    """Read public media metadata with yt-dlp without downloading media or bypassing DRM."""
    from yt_dlp import YoutubeDL

    target = _public_url(url)

    def _run() -> dict[str, Any]:
        with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}) as ydl:
            info = ydl.extract_info(target, download=False)
            clean = ydl.sanitize_info(info)
        keep = {
            "id", "title", "description", "duration", "timestamp", "upload_date", "uploader", "channel",
            "webpage_url", "original_url", "extractor", "view_count", "like_count", "comment_count", "thumbnail",
        }
        return {k: clean.get(k) for k in keep if k in clean}

    metadata = await asyncio.to_thread(_run)
    if metadata.get("description"):
        metadata["description"] = _trim(str(metadata["description"]), 5000)
    return {"ok": True, "metadata": metadata}


@mcp.tool
async def ffmpeg_probe(url: str) -> dict[str, Any]:
    """Inspect a public audio/video URL with FFprobe and return stream/container metadata."""
    target = _public_url(url)

    def _run() -> dict[str, Any]:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", target],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(_trim(proc.stderr, 4000))
        return json.loads(proc.stdout)

    return {"ok": True, "url": target, "probe": await asyncio.to_thread(_run)}


@mcp.tool
def business_tool_routes() -> dict[str, Any]:
    """Return the recommended open-source tool route for each of the user's businesses."""
    return {
        "BidSathi": ["searxng_search", "crawl4ai_extract", "playwright_extract", "Docling worker", "Supabase"],
        "News pages": ["rsshub_fetch", "searxng_search", "playwright_extract", "ytdlp_metadata", "ffmpeg_probe", "Whisper worker"],
        "Founder Features": ["crawl4ai_extract", "playwright_extract", "Supabase CRM tables"],
        "MarketBridge": ["crawl4ai_extract", "playwright_extract", "Supabase", "ERPNext definition"],
        "OneKivo": ["Supabase", "GitHub", "Activepieces definition"],
        "AI Biopics": ["ffmpeg_probe", "Whisper worker", "ComfyUI GPU definition"],
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "3000"))
    mcp.run(transport="http", host="0.0.0.0", port=port)

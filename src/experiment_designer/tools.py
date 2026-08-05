"""Tools for the ExpAgent agentic loop.

- search_papers: directed search across Semantic Scholar (primary), DBLP, arXiv
- read_file: read artifact files passed by ResAgent
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ── Rate limits ──────────────────────────────────────────────────

_SEMANTIC_SCHOLAR_MIN_INTERVAL = 2.0   # seconds between requests
_ARXIV_MIN_INTERVAL = 3.5              # seconds
_DBLP_MIN_INTERVAL = 1.0

_last_request_time: dict[str, float] = {}


def _rate_limit(source: str) -> None:
    """Enforce polite spacing between API requests."""
    now = time.monotonic()
    min_interval = {
        "semantic_scholar": _SEMANTIC_SCHOLAR_MIN_INTERVAL,
        "arxiv": _ARXIV_MIN_INTERVAL,
        "dblp": _DBLP_MIN_INTERVAL,
    }.get(source, 1.0)
    elapsed = now - _last_request_time.get(source, 0)
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_time[source] = time.monotonic()


# ── Result types ─────────────────────────────────────────────────


@dataclass
class SearchResult:
    """A single search result from any source."""

    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""  # e.g. "CVPR 2023"
    abstract: str = ""
    url: str = ""
    paper_id: str = ""  # source-specific id (arXiv id, S2 paper id, etc.)
    source: str = ""  # "semantic_scholar" | "dblp" | "arxiv"


# ── Main search function ─────────────────────────────────────────


def search_papers(
    query: str,
    *,
    source: Literal["semantic_scholar", "dblp", "arxiv"] = "semantic_scholar",
    max_results: int = 5,
    year_from: int | None = None,
    year_to: int | None = None,
    venue_filter: str | None = None,
) -> list[SearchResult]:
    """Search for papers with a directed, purposeful query.

    Args:
        query: Search query (be specific, not broad keywords).
        source: Which API to use.
        max_results: Max results to return (1-20).
        year_from: Filter papers published after this year.
        year_to: Filter papers published before this year.
        venue_filter: Only return papers from this venue (e.g. "CVPR", "ICLR").

    Returns:
        List of SearchResult, ordered by relevance.
    """
    max_results = max(1, min(max_results, 20))

    if source == "semantic_scholar":
        return _search_semantic_scholar(query, max_results, year_from, year_to, venue_filter)
    elif source == "dblp":
        return _search_dblp(query, max_results, year_from, year_to)
    elif source == "arxiv":
        return _search_arxiv(query, max_results, year_from, year_to)
    else:
        raise ValueError(f"Unknown source: {source}")


def save_paper(
    paper_id: str,
    title: str,
    first_author: str = "",
    year: int | None = None,
    abstract: str = "",
    url: str = "",
    code_url: str = "",
    one_liner: str = "",
    output_dir: str = "papers",
) -> str:
    """Save a paper's metadata to disk for later on-demand reading.

    Creates papers/{slug}.md with full metadata. The paper_index in context
    keeps a lightweight reference; the LLM calls read_file when it needs details.
    """
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", title.strip()).strip("_")[:80] or paper_id
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    filepath = out / f"{slug}.md"

    parts = [
        f"# {title}",
        "",
        f"- **Paper ID**: {paper_id}",
        f"- **Authors**: {first_author} et al." if first_author else "",
        f"- **Year**: {year}" if year else "",
        f"- **URL**: {url}" if url else "",
        f"- **Code**: {code_url}" if code_url else "",
        "",
        "## Why this paper matters",
        one_liner or "(no summary provided)",
        "",
        "## Abstract",
        abstract or "(no abstract available)",
    ]
    filepath.write_text("\n".join(parts), encoding="utf-8")

    return (
        f"Saved: {title}\n"
        f"  Paper ID: {paper_id}\n"
        f"  File: {filepath}\n"
        f"  Read full details with read_file(\"{filepath}\")"
    )


def read_file(path: str, max_chars: int = 16_000) -> str:
    """Read a local artifact file (experiment result, log, etc.).

    Args:
        path: Absolute or relative path to the file.
        max_chars: Maximum characters to read (tail of file if exceeded).

    Returns:
        File content as string, truncated from the end if too long.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"[ERROR: file not found: {p}]"
    if not p.is_file():
        return f"[ERROR: not a file: {p}]"

    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[ERROR reading file: {e}]"

    if len(text) > max_chars:
        head = text[:500]
        tail = text[-max_chars + 500:]
        return f"[File: {p} ({len(text)} chars total — showing head 500 + tail {max_chars - 500})]\n{head}\n...\n{tail}"

    return f"[File: {p} ({len(text)} chars)]\n{text}"


# ── Backend implementations ──────────────────────────────────────


def _search_semantic_scholar(
    query: str,
    max_results: int,
    year_from: int | None,
    year_to: int | None,
    venue_filter: str | None,
) -> list[SearchResult]:
    """Search Semantic Scholar Academic Graph API.

    Free tier: ~100 requests per 5 minutes. No API key needed for basic search.
    Docs: https://api.semanticscholar.org/api-docs/
    """
    params: dict[str, str | int] = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,year,venue,abstract,url,externalIds",
    }
    if year_from:
        params["year"] = f"{year_from}-" + (str(year_to) if year_to else "")
    elif year_to:
        params["year"] = f"-{year_to}"

    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(params)

    _rate_limit("semantic_scholar")
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        # Fallback to arXiv on any error
        return _search_arxiv(query, max_results, year_from, year_to)

    results: list[SearchResult] = []
    for item in data.get("data", []):
        authors = [a.get("name", "") for a in item.get("authors", [])]
        year = item.get("year")
        venue = item.get("venue", "") or ""
        abstract = item.get("abstract", "") or ""

        # Venue filter (client-side)
        if venue_filter and venue_filter.lower() not in venue.lower():
            continue

        external = item.get("externalIds", {}) or {}
        paper_url = item.get("url", "") or ""
        paper_id = external.get("DOI") or external.get("ArXiv") or item.get("paperId", "")

        results.append(SearchResult(
            title=item.get("title", "Untitled"),
            authors=authors,
            year=year,
            venue=venue,
            abstract=abstract,
            url=paper_url,
            paper_id=paper_id,
            source="semantic_scholar",
        ))

    # Fallback to DBLP if no results
    if not results:
        return _search_dblp(query, max_results, year_from, year_to)

    return results[:max_results]


def _search_dblp(
    query: str,
    max_results: int,
    year_from: int | None,
    year_to: int | None,
) -> list[SearchResult]:
    """Search DBLP computer science bibliography.

    DBLP is very stable and supports venue-based filtering natively.
    Docs: https://dblp.org/faq/How+to+use+the+dblp+search+API.html
    """
    # Add venue to query if specified (DBLP doesn't have a separate venue filter in search API)
    url = "https://dblp.org/search/publ/api?" + urllib.parse.urlencode({
        "q": query,
        "h": max_results,
        "format": "json",
    })

    _rate_limit("dblp")
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    import defusedxml.ElementTree as ET

    results: list[SearchResult] = []
    hits = data.get("result", {}).get("hits", {}).get("hit", [])

    for item in hits:
        info = item.get("info", {})
        title = info.get("title", "Untitled")
        venue = info.get("venue", "") or ""
        year = info.get("year")
        if year:
            try:
                year = int(year)
            except (ValueError, TypeError):
                year = None

        if year_from and year and year < year_from:
            continue
        if year_to and year and year > year_to:
            continue

        # Parse authors XML
        authors_xml = info.get("authors", {})
        authors: list[str] = []
        if isinstance(authors_xml, dict):
            author_list = authors_xml.get("author", [])
            if isinstance(author_list, str):
                authors = [author_list]
            elif isinstance(author_list, list):
                authors = [a.get("text", "") if isinstance(a, dict) else str(a) for a in author_list]

        paper_url = info.get("url", "") or ""

        results.append(SearchResult(
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            abstract="",  # DBLP doesn't provide abstracts
            url=paper_url,
            paper_id=paper_url,
            source="dblp",
        ))

    return results[:max_results]


def _search_arxiv(
    query: str,
    max_results: int,
    year_from: int | None,
    year_to: int | None,
) -> list[SearchResult]:
    """Search arXiv via the public API.

    Rate limit: ~1 request per 3 seconds. No API key needed.
    Docs: https://info.arxiv.org/help/api/
    """
    import defusedxml.ElementTree as ET

    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)

    _rate_limit("arxiv")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ExpAgent/0.2"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
    except Exception:
        return []

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []

    results: list[SearchResult] = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        title = (title_el.text or "Untitled").strip().replace("\n", " ") if title_el is not None else "Untitled"

        # Authors
        authors: list[str] = []
        for author_el in entry.findall("atom:author/atom:name", ns):
            if author_el.text:
                authors.append(author_el.text.strip())

        # Abstract
        summary_el = entry.find("atom:summary", ns)
        abstract = (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else ""

        # Year
        published_el = entry.find("atom:published", ns)
        year = None
        if published_el is not None and published_el.text:
            try:
                year = int(published_el.text[:4])
            except ValueError:
                pass

        if year_from and year and year < year_from:
            continue
        if year_to and year and year > year_to:
            continue

        # URL and ID
        id_el = entry.find("atom:id", ns)
        paper_id = id_el.text.strip() if id_el is not None else ""
        # Extract arXiv ID from URL
        arxiv_id = paper_id.split("/abs/")[-1] if "/abs/" in paper_id else paper_id

        results.append(SearchResult(
            title=title,
            authors=authors,
            year=year,
            venue="arXiv",
            abstract=abstract,
            url=paper_id,
            paper_id=arxiv_id,
            source="arxiv",
        ))

    return results[:max_results]

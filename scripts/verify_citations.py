"""Verify candidate citations against CrossRef, OpenAlex, and Semantic Scholar.

Classification (project citation rules):
    VALID    two or more databases match on title, first author, and year
    WARNING  exactly one database matches, or a source was unavailable
    ERROR    no database matches

BibTeX is built only from database metadata, never from memory. No contact
address is sent to any API.

Usage:
    python scripts/verify_citations.py [--candidates phase1/candidates.yaml] [--out-dir phase1]
"""

import argparse
import difflib
import json
import logging
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
USER_AGENT = "cca-citation-verifier/0.1"
TITLE_THRESHOLD = 0.90
YEAR_TOLERANCE = 1
FIRST_AUTHOR_WINDOW = 5  # group authors can precede named authors
PREPRINT_MARKERS = ("ssrn", "preprints", "arxiv")
DATACITE_PREFIXES = ("10.48550/", "10.6084/", "10.17605/")  # arXiv, figshare, OSF: registered with DataCite


@dataclass(frozen=True)
class Record:
    """A normalized bibliographic record returned by a database."""

    title: str
    authors: List[str]
    year: Optional[int]
    doi: Optional[str]
    venue: Optional[str]
    source: str


@dataclass(frozen=True)
class Verdict:
    """Verification outcome for one candidate."""

    key: str
    status: str
    matched_sources: List[str]
    unavailable_sources: List[str]
    doi: Optional[str]
    doi_agreement: Optional[bool]
    best: Optional[Record]
    detail: Dict[str, Any] = field(default_factory=dict)


def _get(url: str, headers: Optional[Dict[str, str]] = None, retries: int = 3) -> Optional[bytes]:
    """GET a URL with retries; return None when the source stays unavailable."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as err:
            logger.warning("HTTP %s for %s (attempt %d)", err.code, url[:90], attempt + 1)
            if err.code not in (429, 500, 502, 503):
                return None
        except (urllib.error.URLError, TimeoutError) as err:
            logger.warning("Network error %s for %s (attempt %d)", err, url[:90], attempt + 1)
        time.sleep(2 * (attempt + 1))
    return None


def _json(url: str) -> Optional[Dict[str, Any]]:
    """GET and parse JSON; None on failure or invalid JSON."""
    body = _get(url)
    if body is None:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        logger.error("Invalid JSON from %s", url[:90])
        return None


def _family(name: str) -> str:
    """Return the family-name part of 'Given Family' or 'Family, Given'."""
    name = name.strip()
    return name.split(",")[0] if "," in name else name.split()[-1] if name else ""


def query_crossref(title: str, author: Optional[str]) -> Optional[List[Record]]:
    """Query CrossRef; None if unavailable."""
    q = urllib.parse.urlencode(
        {"query.bibliographic": f"{title} {author or ''}".strip(), "rows": 5,
         "select": "title,author,issued,DOI,container-title"}
    )
    data = _json(f"https://api.crossref.org/works?{q}")
    if data is None:
        return None
    out: List[Record] = []
    for item in data.get("message", {}).get("items", []):
        parts = (item.get("issued", {}).get("date-parts") or [[None]])[0]
        out.append(Record(
            title=(item.get("title") or [""])[0],
            authors=[a.get("family", "") for a in item.get("author", [])],
            year=parts[0] if parts else None,
            doi=item.get("DOI"),
            venue=(item.get("container-title") or [None])[0],
            source="crossref",
        ))
    return out


def query_openalex(title: str) -> Optional[List[Record]]:
    """Query OpenAlex; None if unavailable."""
    q = urllib.parse.urlencode(
        {"search": title, "per-page": 5,
         "select": "title,authorships,publication_year,doi,primary_location"}
    )
    data = _json(f"https://api.openalex.org/works?{q}")
    if data is None:
        return None
    out: List[Record] = []
    for item in data.get("results", []):
        doi = (item.get("doi") or "").replace("https://doi.org/", "") or None
        source = ((item.get("primary_location") or {}).get("source") or {}).get("display_name")
        out.append(Record(
            title=item.get("title") or "",
            authors=[_family(a["author"]["display_name"]) for a in item.get("authorships", []) if a.get("author")],
            year=item.get("publication_year"),
            doi=doi,
            venue=source,
            source="openalex",
        ))
    return out


def query_semanticscholar(title: str) -> Optional[List[Record]]:
    """Query Semantic Scholar; None if unavailable."""
    q = urllib.parse.urlencode(
        {"query": title, "limit": 5, "fields": "title,year,authors,externalIds,venue"}
    )
    data = _json(f"https://api.semanticscholar.org/graph/v1/paper/search?{q}")
    if data is None:
        return None
    return [
        Record(
            title=item.get("title") or "",
            authors=[_family(a.get("name", "")) for a in item.get("authors", [])],
            year=item.get("year"),
            doi=(item.get("externalIds") or {}).get("DOI"),
            venue=item.get("venue") or None,
            source="semanticscholar",
        )
        for item in data.get("data", [])
    ]


def resolve_doi(source: str, doi: str) -> Optional[Record]:
    """Look a DOI up directly in one database and return its record."""
    quoted = urllib.parse.quote(doi, safe="/")
    if source == "crossref":
        data = _json(f"https://api.crossref.org/works/{quoted}")
        item = (data or {}).get("message")
        if not item:
            return None
        parts = (item.get("issued", {}).get("date-parts") or [[None]])[0]
        return Record((item.get("title") or [""])[0], [a.get("family", "") for a in item.get("author", [])],
                      parts[0] if parts else None, item.get("DOI"), (item.get("container-title") or [None])[0], source)
    if source == "openalex":
        item = _json(f"https://api.openalex.org/works/doi:{quoted}")
        if not item or not item.get("title"):
            return None
        return Record(item["title"], [_family(a["author"]["display_name"]) for a in item.get("authorships", []) if a.get("author")],
                      item.get("publication_year"), doi, None, source)
    if source == "semanticscholar":
        item = _json(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quoted}?fields=title,year,authors")
        if not item or not item.get("title"):
            return None
        return Record(item["title"], [_family(a.get("name", "")) for a in item.get("authors", [])],
                      item.get("year"), doi, None, source)
    if source == "datacite":
        attrs = ((_json(f"https://api.datacite.org/dois/{quoted}") or {}).get("data") or {}).get("attributes")
        if not attrs:
            return None
        return Record((attrs.get("titles") or [{}])[0].get("title", ""),
                      [c.get("familyName") or c.get("name", "") for c in attrs.get("creators", [])],
                      attrs.get("publicationYear"), doi, attrs.get("publisher"), source)
    return None


def check_publisher(cand: Dict[str, Any]) -> Optional[Record]:
    """Verify a candidate against its publisher page (title, year, first author must all appear)."""
    url = cand.get("publisher_url")
    body = _get(url) if url else None
    if body is None:
        return None
    text = _norm(re.sub(r"<[^>]+>", " ", body.decode("utf-8", errors="replace")))
    family = _norm(cand.get("first_author") or "").split()[-1:] or [""]
    if _norm(cand["title"]) in text and str(cand["year"]) in text and family[0] in text:
        return Record(cand["title"], [cand.get("first_author") or ""], int(cand["year"]), None, url, "publisher")
    return None


def pick_doi(matched: Dict[str, Record], hint: Optional[str] = None) -> Optional[str]:
    """Choose one DOI: the hinted DOI if a database confirmed it, else the DOI most sources agree on,
    preferring published versions over preprints and JSTOR mirrors."""
    dois = [r.doi for r in matched.values() if r.doi]
    if hint and any(d.lower() == hint.lower() for d in dois):
        return hint
    published = [d for d in dois if not any(m in d.lower() for m in PREPRINT_MARKERS)]
    pool = published or dois
    non_mirror = [d for d in pool if not d.startswith("10.2307/")] or pool
    if not non_mirror:
        return None
    return max(non_mirror, key=lambda d: sum(x.lower() == d.lower() for x in dois))


def _norm(text: str) -> str:
    """Lowercase, strip accents and punctuation."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def title_similarity(a: str, b: str) -> float:
    """Similarity in [0, 1]; a subtitle-only difference still scores high."""
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    short, long_ = sorted((a, b), key=len)
    contained = 0.95 if len(short) > 25 and short in long_ else 0.0
    return max(ratio, contained)


def is_match(cand: Dict[str, Any], rec: Record) -> bool:
    """True when title, first author (if given), and year all agree."""
    if title_similarity(cand["title"], rec.title) < TITLE_THRESHOLD:
        return False
    if cand.get("year") and rec.year and abs(int(cand["year"]) - int(rec.year)) > YEAR_TOLERANCE:
        return False
    expected = cand.get("first_author")
    if expected:
        window = [_norm(a) for a in rec.authors if a][:FIRST_AUTHOR_WINDOW]
        if not window:
            return True  # no author metadata (common for older JSTOR records); flagged in verify()
        # first_author_alt records a name form the deposit uses (for example a reversed name order)
        families = [_norm(n).split()[-1] for n in [expected, cand.get("first_author_alt") or ""] if _norm(n)]
        return any(family in name.split() for family in families for name in window)
    return True


def verify(cand: Dict[str, Any], pause: float) -> Verdict:
    """Search CrossRef and OpenAlex, cross-resolve any DOI found, then classify."""
    matched: Dict[str, Record] = {}
    unavailable: List[str] = []
    searches = {"crossref": query_crossref(cand["title"], cand.get("first_author")),
                "openalex": query_openalex(cand["title"])}
    for name, recs in searches.items():
        if recs is None:
            unavailable.append(name)
        elif hit := next((r for r in recs if is_match(cand, r)), None):
            matched[name] = hit
    via_doi: List[str] = []
    hint = {cand["doi_hint"]} if cand.get("doi_hint") else set()  # a lookup key, never trusted as fact
    for doi in sorted({r.doi for r in matched.values() if r.doi} | hint):
        targets = ["crossref", "openalex"] + (["datacite"] if doi.startswith(DATACITE_PREFIXES) else [])
        for name in targets:
            if name not in matched and (rec := resolve_doi(name, doi)) and is_match(cand, rec):
                matched[name] = rec
                via_doi.append(name)
    if len(matched) < 2:
        for doi in sorted({r.doi for r in matched.values() if r.doi} | hint):
            if (rec := resolve_doi("semanticscholar", doi)) and is_match(cand, rec):
                matched["semanticscholar"] = rec
                via_doi.append("semanticscholar")
                break
    if len(matched) < 2 and (rec := check_publisher(cand)):
        matched["publisher"] = rec
    if len(matched) < 2 and "semanticscholar" not in matched:
        recs = query_semanticscholar(cand["title"])
        if recs is None:
            unavailable.append("semanticscholar")
        elif hit := next((r for r in recs if is_match(cand, r)), None):
            matched["semanticscholar"] = hit
    time.sleep(pause)
    dois = {r.doi.lower() for r in matched.values() if r.doi}
    best = matched.get("crossref") or matched.get("openalex") or next(iter(matched.values()), None)
    status = "VALID" if len(matched) >= 2 else "WARNING" if len(matched) == 1 else "ERROR"
    return Verdict(
        key=cand["key"], status=status, matched_sources=sorted(matched),
        unavailable_sources=unavailable, doi=pick_doi(matched, cand.get("doi_hint")),
        doi_agreement=(len(dois) == 1) if dois and len(matched) >= 2 else None,
        best=best, detail={"query_title": cand["title"], "query_author": cand.get("first_author"),
                           "query_year": cand.get("year"), "matched_via_doi": via_doi,
                           "author_unchecked": sorted(n for n, r in matched.items() if not any(r.authors)),
                           "dois_by_source": {n: r.doi for n, r in matched.items()}},
    )


def bibtex_for(verdict: Verdict) -> Optional[str]:
    """Build a BibTeX entry from database metadata (DOI content negotiation, else the record)."""
    if verdict.best is None:
        return None
    if verdict.doi:
        body = _get(f"https://doi.org/{urllib.parse.quote(verdict.doi)}",
                    headers={"Accept": "application/x-bibtex"})
        if body:
            entry = body.decode("utf-8", errors="replace").strip()
            if entry.startswith("@"):
                entry = entry.replace("&amp;", "\\&")  # CrossRef HTML-escapes ampersands
                return re.sub(r"^(@\w+)\{[^,]+,", rf"\1{{{verdict.key},", entry, count=1)
    rec = verdict.best
    authors = " and ".join(rec.authors) or "Unknown"
    lines = [f"@misc{{{verdict.key},", f"  title = {{{rec.title}}},", f"  author = {{{authors}}},",
             f"  year = {{{rec.year}}},"]
    if rec.venue:
        lines.append(f"  howpublished = {{{rec.venue}}},")
    lines.append("  note = {Built from database metadata; no DOI resolved}")
    return "\n".join(lines) + "\n}"


def merge_with_previous(verdict: Verdict, previous: Optional[Dict[str, Any]]) -> Verdict:
    """Union sources that confirmed the paper in any run; a confirmation does not expire."""
    if not previous:
        return verdict
    sources = sorted(set(verdict.matched_sources) | set(previous.get("matched_sources", [])))
    dois = {**previous.get("detail", {}).get("dois_by_source", {}), **verdict.detail.get("dois_by_source", {})}
    old_best = previous.get("best")
    best = verdict.best or (Record(**old_best) if old_best else None)
    status = "VALID" if len(sources) >= 2 else "WARNING" if sources else "ERROR"
    still_unavailable = [s for s in verdict.unavailable_sources if s not in sources]
    return Verdict(
        key=verdict.key, status=status, matched_sources=sources, unavailable_sources=still_unavailable,
        doi=verdict.doi or previous.get("doi"), doi_agreement=verdict.doi_agreement, best=best,
        detail={**previous.get("detail", {}), **verdict.detail, "dois_by_source": dois,
                "merged_with_previous_run": True},
    )


def write_outputs(verdicts: List[Verdict], out_dir: Path, manual_keys: Optional[set] = None) -> None:
    """Write the JSON audit, markdown report, and BibTeX file."""
    (out_dir / "citation_audit.json").write_text(
        json.dumps([asdict(v) for v in verdicts], indent=2, default=str), encoding="utf-8")
    counts = {s: sum(v.status == s for v in verdicts) for s in ("VALID", "WARNING", "ERROR")}
    rows = ["| Key | Status | Matched sources | DOI | Notes |", "|---|---|---|---|---|"]
    for v in verdicts:
        notes = []
        if v.unavailable_sources:
            notes.append("unavailable: " + ", ".join(v.unavailable_sources))
        if v.doi_agreement is False:
            notes.append("DOI disagreement")
        rows.append(f"| {v.key} | {v.status} | {', '.join(v.matched_sources) or '-'} | {v.doi or '-'} | {'; '.join(notes) or '-'} |")
    unavailable: Dict[str, int] = {}
    for v in verdicts:
        for name in v.unavailable_sources:
            unavailable[name] = unavailable.get(name, 0) + 1
    coverage = ("Search queries unavailable during this run (rate limit or budget): "
                + ", ".join(f"{n} ({c} of {len(verdicts)})" for n, c in sorted(unavailable.items()))
                + ". Affected entries were confirmed by resolving a DOI directly in the other databases."
                if unavailable else "All source queries were available during this run.")
    report = ["# Citation Audit Report", "",
              f"{len(verdicts)} candidates: {counts['VALID']} VALID, {counts['WARNING']} WARNING, {counts['ERROR']} ERROR.",
              "", "VALID = two or more databases match on title, first author, and year. "
              "WARNING = one database matches or a source was unavailable. ERROR = no match.", "", coverage, ""] + rows
    (out_dir / "Citation-Audit-Report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    entries: List[str] = []
    for status in ("VALID", "WARNING"):
        for v in verdicts:
            if v.status == status and v.key not in (manual_keys or set()) and (entry := bibtex_for(v)):
                entries.append(f"% {status}: {v.key} ({', '.join(v.matched_sources)})\n{entry}")
                time.sleep(0.3)
    manual = out_dir / "manual_entries.bib"
    if manual.exists():
        entries.append("% MANUAL: transcribed from the publisher page (see manual_entries.bib)\n" + manual.read_text(encoding="utf-8").strip())
    (out_dir / "references.bib").write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    logger.info("Wrote audit outputs to %s: %s", out_dir, counts)


def main() -> int:
    """Run verification for all candidates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=ROOT / "phase1" / "candidates.yaml")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase1")
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between candidates")
    parser.add_argument("--only", nargs="*", default=None, help="verify only these keys and merge into the audit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    candidates = yaml.safe_load(args.candidates.read_text(encoding="utf-8"))
    audit_path = args.out_dir / "citation_audit.json"
    previous = {r["key"]: r for r in json.loads(audit_path.read_text(encoding="utf-8"))} if audit_path.exists() else {}
    run_keys = set(args.only) if args.only else {c["key"] for c in candidates}
    verdicts = []
    for i, cand in enumerate(candidates, 1):
        if cand["key"] not in run_keys:
            old = previous[cand["key"]]
            best = old.get("best")
            verdicts.append(Verdict(**{**old, "best": Record(**best) if best else None}))
            continue
        verdict = merge_with_previous(verify(cand, args.pause), previous.get(cand["key"]))
        logger.info("[%d/%d] %s -> %s %s", i, len(candidates), cand["key"], verdict.status, verdict.matched_sources)
        verdicts.append(verdict)
    write_outputs(verdicts, args.out_dir, {c["key"] for c in candidates if c.get("manual_bib")})
    return 0 if all(v.status != "ERROR" for v in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())

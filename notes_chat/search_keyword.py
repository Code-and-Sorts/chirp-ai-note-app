from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from utils.file_utils import NoteRecord, list_notes

EXCERPT_WIDTH = 120
MAX_EXCERPTS_PER_NOTE = 5
RULE_WIDTH = 47
HORIZONTAL_RULE = "─" * RULE_WIDTH


@dataclass(frozen=True)
class SearchOptions:
    query: str
    since_minutes: int | None = None
    regex: bool = False
    json: bool = False


@dataclass
class _Excerpt:
    source: str
    line: int
    text: str
    span: tuple[int, int]


@dataclass
class _NoteHit:
    record: NoteRecord
    note_id: int
    title: str
    excerpts: list[_Excerpt] = field(default_factory=list)

    @property
    def hits(self) -> int:
        return len(self.excerpts)


def run_search(settings: Any, options: SearchOptions) -> dict:
    notes_root = settings.directories.notes_root
    pattern = _compile_pattern(options)

    records = [
        record
        for record in list_notes(notes_root)
        if record.transcript is not None or record.notes is not None
    ]
    total_notes_scanned = len(records)
    records = _apply_since(records, options.since_minutes)

    started = time.perf_counter()

    hits: list[_NoteHit] = []
    newest_first = list(reversed(records))
    for note_id, record in enumerate(newest_first, start=1):
        excerpts: list[_Excerpt] = []
        if record.transcript is not None:
            excerpts.extend(_extract_matches(record.transcript, "transcript", pattern))
        if record.notes is not None:
            excerpts.extend(_extract_matches(record.notes, "notes", pattern))
        if not excerpts:
            continue
        title = _resolve_title(record)
        hits.append(
            _NoteHit(
                record=record,
                note_id=note_id,
                title=title,
                excerpts=excerpts[:MAX_EXCERPTS_PER_NOTE],
            )
        )

    elapsed = time.perf_counter() - started
    hits.sort(key=lambda h: (-h.hits, h.note_id))

    return {
        "query": options.query,
        "since": _since_to_text(options.since_minutes),
        "regex": options.regex,
        "duration_seconds": round(elapsed, 2),
        "total_notes_scanned": total_notes_scanned,
        "matches": [_hit_to_dict(hit) for hit in hits],
    }


def render_results(console: Console, options: SearchOptions, result: dict) -> None:
    query = options.query
    total = result["total_notes_scanned"]
    matches = result["matches"]

    console.print()
    console.print(
        f" [dim]searching {total} {_plural(total, 'note')} for[/dim] "
        f'[bold yellow]"{query}"[/bold yellow] '
        f"[dim]· keyword (ripgrep over transcripts + notes)[/dim]"
    )
    console.print()

    if options.since_minutes is not None:
        console.print(
            f" [dim]scope: last {_humanize_duration(options.since_minutes)}[/dim]"
        )
        console.print()

    if not matches:
        return

    note_count = len(matches)
    match_count = sum(m["hits"] for m in matches)
    console.print(
        f" [bold white]{note_count} {_plural(note_count, 'note')}[/bold white] "
        f"[dim]· {match_count} {_plural(match_count, 'match', 'matches')} "
        f"· {result['duration_seconds']:.2f}s[/dim]"
    )
    console.print()

    table = Table(
        show_header=True,
        header_style="yellow bold",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("#", style="dim", no_wrap=True, justify="right")
    table.add_column("title", style="white")
    table.add_column("date", style="cyan", no_wrap=True)
    table.add_column("hits", style="bold orange3", no_wrap=True, justify="right")

    for match in matches:
        table.add_row(
            str(match["id"]),
            match["title"],
            _format_date(match["date"]),
            str(match["hits"]),
        )
    console.print(table)
    console.print()

    for match in matches:
        console.print(
            f" [bold yellow]› #{match['id']} {match['title']}[/bold yellow] "
            f"[dim]· {_format_date(match['date'])}[/dim]"
        )
        for excerpt in match["excerpts"]:
            prefix = (
                "transcript"
                if excerpt["source"] == "transcript"
                else f"notes.md:{excerpt['line']}"
            )
            text = _markup_excerpt(excerpt["text"], options.query, options.regex)
            console.print(f"   [dim]{prefix}[/dim]  {text}")
        console.print()

    top_id = matches[0]["id"]
    console.print(f" [dim]› chirp notes view {top_id}      · open a result[/dim]")
    console.print(
        f' [dim]› chirp ask "{query}"          · semantic answer instead[/dim]'
    )


def render_no_matches(
    console: Console,
    options: SearchOptions,
    total_scanned: int,
    suggestions: list[tuple[str, int]],
) -> None:
    query = options.query
    console.print()
    console.print(
        f" [dim]searching {total_scanned} {_plural(total_scanned, 'note')} for[/dim] "
        f'[bold yellow]"{query}"[/bold yellow]'
    )
    console.print()
    console.print(" [red]✗[/red] [bold white]no exact matches.[/bold white]")
    console.print()
    console.print(f" [dim]{HORIZONTAL_RULE}[/dim]")
    console.print(
        " [bold yellow]![/bold yellow] keyword search is literal. for fuzzy/semantic"
    )
    console.print("   answers, try chirp ask:")
    console.print()
    console.print(f'   [dim]$[/dim] chirp ask "{query}"')

    if suggestions:
        console.print()
        console.print(" [dim]close keywords found across your notes:[/dim]")
        for token, doc_count in suggestions:
            label = f'"{token}"'
            console.print(
                f"   [dim]·[/dim] {label}     [dim]({doc_count} {_plural(doc_count, 'note')})[/dim]"
            )


def suggest_close_keywords(bm25_path: Path, query: str) -> list[tuple[str, int]]:
    try:
        from notes_chat.bm25 import BM25Index
    except ImportError:
        return []
    if not bm25_path.exists():
        return []

    try:
        index = BM25Index(bm25_path)
        vocab = index.vocabulary()
    except Exception:
        return []
    if not vocab:
        return []

    query_tokens = [tok for tok in re.findall(r"\w+", query.lower()) if len(tok) >= 2]
    if not query_tokens:
        return []

    candidates: dict[str, int] = {}
    for token, doc_count in vocab.items():
        if token in query_tokens:
            continue
        if any(_is_close(token, qt) for qt in query_tokens):
            candidates[token] = doc_count

    ranked = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:3]


def _compile_pattern(options: SearchOptions) -> re.Pattern[str]:
    if options.regex:
        try:
            return re.compile(options.query, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc.msg}") from exc
    return re.compile(re.escape(options.query), re.IGNORECASE)


def _apply_since(
    records: list[NoteRecord], since_minutes: int | None
) -> list[NoteRecord]:
    if since_minutes is None:
        return records
    cutoff = datetime.now() - timedelta(minutes=since_minutes)
    return [r for r in records if r.created_at >= cutoff]


def _extract_matches(
    path: Path, source: str, pattern: re.Pattern[str]
) -> list[_Excerpt]:
    excerpts: list[_Excerpt] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return excerpts

    for line_no, line in enumerate(text.splitlines(), start=1):
        match = pattern.search(line)
        if match is None:
            continue
        excerpts.append(
            _Excerpt(
                source=source,
                line=line_no,
                text=_window_excerpt(line, match.start(), match.end()),
                span=(match.start(), match.end()),
            )
        )
    return excerpts


def _window_excerpt(line: str, match_start: int, match_end: int) -> str:
    line = line.rstrip()
    if len(line) <= EXCERPT_WIDTH:
        return line

    match_len = match_end - match_start
    side_budget = max(0, (EXCERPT_WIDTH - match_len - 2) // 2)
    start = max(0, match_start - side_budget)
    end = min(len(line), match_end + side_budget)

    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(line) else ""
    return f"{prefix}{line[start:end]}{suffix}"


def _markup_excerpt(text: str, query: str, regex: bool) -> str:
    pattern = re.compile(query if regex else re.escape(query), re.IGNORECASE)
    out: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        out.append(text[cursor : match.start()])
        out.append(f"[bold yellow]{match.group(0)}[/bold yellow]")
        cursor = match.end()
    out.append(text[cursor:])
    return "".join(out)


def _hit_to_dict(hit: _NoteHit) -> dict:
    return {
        "id": hit.note_id,
        "slug": hit.record.slug,
        "title": hit.title,
        "date": hit.record.created_at.date().isoformat(),
        "hits": hit.hits,
        "excerpts": [
            {"source": e.source, "line": e.line, "text": e.text} for e in hit.excerpts
        ],
    }


def _resolve_title(record: NoteRecord) -> str:
    if record.title:
        return record.title
    return record.slug


def _format_date(iso_date: str) -> str:
    try:
        dt = datetime.strptime(iso_date, "%Y-%m-%d")
    except ValueError:
        return iso_date
    return dt.strftime("%b %d").lower()


def _humanize_duration(minutes: int) -> str:
    weeks, rem = divmod(minutes, 7 * 24 * 60)
    days, rem = divmod(rem, 24 * 60)
    hours = rem // 60

    if weeks and not days and not hours:
        return f"{weeks} {_plural(weeks, 'week')}"
    if days and not weeks and not hours:
        return f"{days} {_plural(days, 'day')}"
    if hours and not weeks and not days:
        return f"{hours} {_plural(hours, 'hour')}"
    if minutes < 60:
        return f"{minutes} {_plural(minutes, 'minute')}"

    total_days = minutes // (24 * 60)
    if total_days > 0:
        return f"{total_days} {_plural(total_days, 'day')}"
    total_hours = max(1, minutes // 60)
    return f"{total_hours} {_plural(total_hours, 'hour')}"


def _since_to_text(minutes: int | None) -> str | None:
    if minutes is None:
        return None
    return _humanize_duration(minutes)


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    if count == 1:
        return singular
    return plural if plural is not None else f"{singular}s"


def _is_close(candidate: str, query_token: str) -> bool:
    if not candidate or not query_token:
        return False
    if len(query_token) >= 3 and query_token in candidate:
        return True
    if len(candidate) >= 3 and candidate in query_token:
        return True
    if len(candidate) >= 3 and len(query_token) >= 3:
        if candidate[:3] == query_token[:3]:
            return True
    return _levenshtein(candidate, query_token) <= 2


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if abs(len(a) - len(b)) > 2:
        return 3
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def emit_json(result: dict) -> str:
    return json.dumps(result, indent=2)

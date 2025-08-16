#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developer Cleaning Script for Parsed EHR TXT Files (standalone)
===============================================================

Purpose
-------
Quick, config-free cleaner for already **parsed** EHR .txt files.
Reads from --input_dir (e.g., data/processed/parsed_txt) and writes
cleaned files to --output_dir (e.g., data/processed/cleaned_txt),
plus an audit CSV with a TOTAL summary row.

Typical Run
-----------
python src/preprocessing/dev_clean_ehr_text.py \
  --input_dir data/processed/parsed_txt \
  --output_dir data/processed/cleaned_txt \
  --overwrite False

Key Behaviors
-------------
- Removes placeholder lines (e.g., "Extraction failed")
- Drops trailing empty key-value tails like:  "| Dose:"  "| Status:"  ...
- Trims + collapses whitespace; removes spaces before punctuation
- Normalizes dates to YYYY-MM-DD
- Rounds long decimals (>= 3 decimals) to 2 decimals
- De-duplicates identical lines within a file
- Writes '\n' newlines deterministically and ensures a trailing newline
- Safety guard: refuses to run if input==output or input under data/raw/

Extra Flags
-----------
--limit N       : process only first N files (dev sanity check)
--dry_run True  : don't write outputs; just print stats and produce report
--report_name   : name of the CSV report (default: cleaning_report.csv)

This script is intentionally standalone (no project config import).
"""

from __future__ import annotations
import re
import csv
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# ----------- Regexes & Rules -----------
DATE_RE  = re.compile(
    r'(?P<full>(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})(?:[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:\d{2})?)?)'
)
LONG_RE  = re.compile(r'(?P<num>\d+\.\d{3,})')  # long decimals -> round to 2
PLACEH   = [
    re.compile(r'Extraction failed', re.IGNORECASE),
]
# Drop trailing empty key-value tails like: "| Dose:" or "| Interpretation:"
EMPTY_T  = re.compile(
    r'\|\s*(Dose|Status|Interpretation|Value|Result|Conclusion|Intent|Date)\s*:\s*$'
)
MSPACE   = re.compile(r'\s+')
SBPUNCT  = re.compile(r'\s+([,;:])')
# Remove zero-width or stray control characters (safe subset)
CTRL_CHARS = re.compile(r'[\u200b\u200c\u200d\uFEFF]')

PERIOD_RE = re.compile(
    r'(?:^|\b)Period:\s*(?P<start>\d{4}-\d{1,2}-\d{1,2})(?:\s*to\s*(?P<end>\d{4}-\d{1,2}-\d{1,2}))?',
    re.IGNORECASE
)

# a Date segment must be YYYY-MM-DD; anything else gets removed
BAD_DATE_SEG = re.compile(r'(\s*\|\s*)?Date\s*:\s*(?!\d{4}-\d{2}-\d{2}(\b|$)).*?(?=\s*\||$)', re.IGNORECASE)# Normalize YYYY-MM-DD components


def _norm_ymd(y, m, d):
    try:
        y, m, d = int(y), int(m), int(d)
        if 1 <= m <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{m:02d}-{d:02d}"
    except:
        pass
    return None

def normalize_periods(s: str) -> str:
    """Normalize 'Period: YYYY-MM-DD to YYYY-MM-DD' or drop incomplete end dates."""
    def _r(m):
        start = m.group('start')
        end = m.group('end')
        # Normalize components
        sy, sm, sd = start.split('-')
        start_n = _norm_ymd(sy, sm, sd) or start

        if end:
            ey, em, ed = end.split('-')
            end_n = _norm_ymd(ey, em, ed)
            if not end_n:
                # Incomplete end → keep start only
                return f"Period: {start_n}"
            # If end < start, swap
            if end_n < start_n:
                start_n, end_n = end_n, start_n
            return f"Period: {start_n} to {end_n}"
        else:
            return f"Period: {start_n}"
    return PERIOD_RE.sub(_r, s)

def replace_units(s: str) -> str:
    # ' Cel' → ' °C'
    s = re.sub(r'\bCel\b', '°C', s)
    # optional: normalize 'kg/m2' -> 'kg/m²'
    s = re.sub(r'kg/m2\b', 'kg/m²', s)
    return s


def strip_malformed_date_segments(s: str, stats: dict) -> str:
    before = s
    s = BAD_DATE_SEG.sub('', s)           # drop bad "Date: …"
    if s != before:
        stats['bad_date_segments'] = stats.get('bad_date_segments', 0) + 1
        # tidy pipes & spaces if we removed a middle/end segment
        s = re.sub(r'\s*\|\s*\|\s*', ' | ', s)
        s = re.sub(r'^\s*\|\s*|\s*\|\s*$', '', s).strip()
        s = re.sub(r'\s{2,}', ' ', s)
    return s


def strip_empty_kv_segments(s: str, stats: dict) -> str:
    """
    For segments separated by ' | ', remove key:value pairs where value is empty.
    We keep plain segments (no colon) and non-empty key:value segments.
    """
    parts = [p.strip() for p in s.split('|')]
    kept = []
    removed = 0
    for seg in parts:
        if ':' not in seg:
            if seg: kept.append(seg)
            continue
        key, val = [x.strip() for x in seg.split(':', 1)]
        if val == "":
            removed += 1
            continue
        kept.append(f"{key}: {val}")
    if removed:
        stats['empty_kv_removed'] = stats.get('empty_kv_removed', 0) + removed
    return " | ".join(kept).strip()

# ----------- Cleaning helpers -----------
def normalize_date(s: str, stats: Dict[str, int]) -> str:
    def _r(m: re.Match) -> str:
        try:
            y, mm, dd = int(m.group('y')), int(m.group('m')), int(m.group('d'))
            if 1 <= mm <= 12 and 1 <= dd <= 31:
                stats['dates_normalized'] += 1
                return f"{y:04d}-{mm:02d}-{dd:02d}"
        except Exception:
            pass
        return m.group('full')
    return DATE_RE.sub(_r, s)

def round_long(s: str, stats: Dict[str, int]) -> str:
    def _r(m: re.Match) -> str:
        stats['numbers_rounded'] += 1
        return f"{float(m.group('num')):.2f}"
    return LONG_RE.sub(_r, s)

def clean_line(line: str, stats: dict):
    # Drop placeholder lines
    for p in PLACEH:
        if p.search(line):
            stats['dropped_placeholders'] += 1
            return None

    # Trim & basic empties
    line = line.strip()
    if not line:
        stats['empty_lines_removed'] += 1
        return None

    # Collapse spaces & punctuation spacing
    line = SBPUNCT.sub(r'\1', line)
    line2 = MSPACE.sub(' ', line)
    if line2 != line:
        stats['whitespace_collapsed'] += 1
    line = line2

    # Normalize dates & numbers
    line = normalize_date(line, stats)
    line = round_long(line, stats)

    # Normalize 'Period' blocks and units
    line = normalize_periods(line)
    line = replace_units(line)

     # NEW: remove malformed Date: fields (e.g., "Date: 2011-" or "Date: 2016")
    line = strip_malformed_date_segments(line, stats)

    # Then strip truly empty KV segments anywhere
    line = strip_empty_kv_segments(line, stats)



    # Remove trailing empty-tail lines like '| Dose:' that appear ALONE at EOL
    if EMPTY_T.search(line):
        stats['dropped_empty_tails'] += 1
        return None

    # Finally, strip empty key:value segments inside the line (Dose:, Intent:, Interpretation:, Date: when blank)
    line = strip_empty_kv_segments(line, stats)

    # If the line became empty after stripping, drop it
    if not line:
        stats['empty_lines_removed'] += 1
        return None

    return line

def dedupe(lines: List[str], stats: Dict[str, int]) -> List[str]:
    seen, out = set(), []
    for l in lines:
        if l in seen:
            stats['duplicates_removed'] += 1
        else:
            seen.add(l)
            out.append(l)
    return out

# ----------- Safety -----------
def assert_safety(inp: Path, out: Path) -> None:
    if inp.resolve() == out.resolve():
        raise RuntimeError("Refusing to run: input_dir == output_dir")
    raw = Path("data/raw").resolve()
    if raw == inp.resolve() or raw in inp.resolve().parents:
        raise RuntimeError("Refusing to run: input_dir is inside data/raw/")

# ----------- IO helpers -----------
def read_text_lines(path: Path) -> List[str]:
    """
    Read text robustly. Try UTF-8, then fall back to latin-1 if needed.
    Normalize line endings to '\n' and drop trailing '\r'.
    """
    try:
        txt = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        txt = path.read_text(encoding='latin-1')
    return txt.splitlines()

def write_text(path: Path, content: str) -> None:
    """
    Write using UTF-8, deterministic '\n' newlines, ensure final trailing newline.
    """
    if content and not content.endswith("\n"):
        content = content + "\n"
    # Explicit newline enforcement
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    path.write_text(content, encoding='utf-8')

# ----------- Per-file processing -----------
def process_file(in_p: Path, out_p: Path, overwrite: bool, dry_run: bool) -> Dict[str, object]:
    s: Dict[str, object] = dict(
        file=in_p.name, status='ok', original_lines=0, kept_lines=0,
        dropped_placeholders=0, dropped_empty_tails=0, empty_lines_removed=0,
        whitespace_collapsed=0, numbers_rounded=0, dates_normalized=0,
        duplicates_removed=0
    )

    if out_p.exists() and not overwrite:
        s['status'] = 'skipped_exists'
        return s

    try:
        lines = read_text_lines(in_p)
        s['original_lines'] = len(lines)

        cleaned: List[str] = []
        for ln in lines:
            x = clean_line(ln, s)  # type: ignore[arg-type]
            if x is not None:
                cleaned.append(x)

        cleaned = dedupe(cleaned, s)  # type: ignore[arg-type]
        s['kept_lines'] = len(cleaned)

        if not dry_run:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            write_text(out_p, "\n".join(cleaned))

    except Exception as e:
        s['status'] = f"error:{e}"
    return s

# ----------- Report -----------
def write_report_with_total(path: Path, rows: List[Dict[str, object]]) -> None:
    hdr = [
    'file','status','original_lines','kept_lines','dropped_placeholders',
    'dropped_empty_tails','empty_lines_removed','whitespace_collapsed',
    'numbers_rounded','dates_normalized','duplicates_removed',
    'bad_date_segments'  # <— add
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in hdr})

        def S(k: str) -> int:
            total = 0
            for r in rows:
                v = r.get(k, 0)
                try:
                    total += int(v)  # type: ignore[arg-type]
                except Exception:
                    total += 0
            return total

        total_row = {
            'file':'TOTAL',
            'status':'',
            'original_lines': S('original_lines'),
            'kept_lines': S('kept_lines'),
            'dropped_placeholders': S('dropped_placeholders'),
            'dropped_empty_tails': S('dropped_empty_tails'),
            'empty_lines_removed': S('empty_lines_removed'),
            'whitespace_collapsed': S('whitespace_collapsed'),
            'numbers_rounded': S('numbers_rounded'),
            'dates_normalized': S('dates_normalized'),
            'duplicates_removed': S('duplicates_removed'),
        }
        w.writerow(total_row)

# ----------- CLI -----------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Clean processed EHR TXT files (standalone, no config).")
    ap.add_argument('--input_dir',  required=True, help='Directory with parsed .txt files.')
    ap.add_argument('--output_dir', required=True, help='Directory to write cleaned .txt files.')
    ap.add_argument('--overwrite',  type=lambda x: str(x).lower() in {'1','true','yes'}, default=False,
                    help='Overwrite existing outputs (default: False).')
    ap.add_argument('--limit',      type=int, default=0,
                    help='Process only the first N files (0 = no limit).')
    ap.add_argument('--dry_run',    type=lambda x: str(x).lower() in {'1','true','yes'}, default=False,
                    help='If True, do not write cleaned files (report still generated).')
    ap.add_argument('--report_dir',  type=str, default='reports',
                help='Directory where the audit CSV will be written (default: reports)')
    ap.add_argument('--report_name', type=str, default='cleaning_report.csv',
                    help='Name of the CSV report written to output_dir (default: cleaning_report.csv).')
    return ap.parse_args()



def main() -> None:
    args = parse_args()
    inp, out = Path(args.input_dir), Path(args.output_dir)

    if not inp.exists():
        print(f"[ERROR] input_dir not found: {inp}", file=sys.stderr)
        sys.exit(1)

    assert_safety(inp, out)

    files = sorted(p for p in inp.glob("*.txt") if p.is_file())
    if args.limit and args.limit > 0:
        files = files[:args.limit]

    if not files:
        print(f"[WARN] No .txt files found in {inp}", file=sys.stderr)

    rows = []
    for p in files:
        s = process_file(p, out / p.name, overwrite=args.overwrite, dry_run=args.dry_run)
        rows.append(s)
        drop = int(s['dropped_placeholders']) + int(s['dropped_empty_tails']) + int(s['empty_lines_removed'])
        print(f"[{s['status']:>12}] {p.name} "
              f"(orig={s['original_lines']} -> kept={s['kept_lines']}, drop={drop}, dups={s['duplicates_removed']})")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / args.report_name

    write_report_with_total(report_path, rows)
    print(f"\n✔ Done. Report (with TOTAL): {report_path}")

if __name__ == "__main__":
    main()

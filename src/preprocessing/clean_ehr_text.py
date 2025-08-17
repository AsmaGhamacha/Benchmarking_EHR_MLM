#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean already-parsed EHR .txt files -> cleaned_txt
- Reads project paths from config.yaml
- Input:  paths.parsed_txt
- Output: paths.cleaned_txt
- Audit:  cleaned_txt/cleaning_report.csv (+ TOTAL summary row)
- Safety: refuses to touch data/raw or clean in-place

Run:
  python -m src.preprocessing.clean_ehr_text
  # or override:
  python -m src.preprocessing.clean_ehr_text --input_dir data/processed/parsed_txt --output_dir data/processed/cleaned_txt --overwrite False
"""

import os
import re
import csv
import sys
import argparse
from pathlib import Path

try:
    import yaml  # PyYAML
except ImportError as e:
    print("[ERROR] PyYAML not installed. Add `pyyaml` to requirements.txt and pip install.", file=sys.stderr)
    raise

# ----------- Regexes & Rules -----------
DATE_RE = re.compile(
    r'(?P<full>(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})(?:[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:\d{2})?)?)'
)
LONG_FLOAT_RE = re.compile(r'(?P<num>\d+\.\d{3,})')
PLACEHOLDER_PATTERNS = [re.compile(r'Extraction failed', re.IGNORECASE)]
EMPTY_TAIL_RE = re.compile(r'\|\s*(Dose|Status|Interpretation|Value|Result|Conclusion)\s*:\s*$')
MULTISPACE_RE = re.compile(r'\s+')
SPACE_BEFORE_PUNCT_RE = re.compile(r'\s+([,;:])')

# ----------- Config helpers -----------
def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_default_paths_from_config(cfg: dict) -> tuple[Path, Path]:
    paths = cfg.get("paths", {})
    parsed = paths.get("parsed_txt", "data/processed/parsed_txt")
    cleaned = paths.get("cleaned_txt", "data/processed/cleaned_txt")
    return Path(parsed), Path(cleaned)

# ----------- Cleaning funcs -----------
def normalize_date_in_line(line: str, stats: dict) -> str:
    def _repl(m):
        try:
            y = int(m.group('y')); mm = int(m.group('m')); dd = int(m.group('d'))
            if 1 <= mm <= 12 and 1 <= dd <= 31:
                stats['dates_normalized'] += 1
                return f"{y:04d}-{mm:02d}-{dd:02d}"
        except Exception:
            pass
        return m.group('full')
    return DATE_RE.sub(_repl, line)

def round_long_floats(line: str, stats: dict) -> str:
    def _repl(m):
        stats['numbers_rounded'] += 1
        return f"{float(m.group('num')):.2f}"
    return LONG_FLOAT_RE.sub(_repl, line)

def clean_line(line: str, stats: dict) -> str | None:
    # Drop placeholders
    for pat in PLACEHOLDER_PATTERNS:
        if pat.search(line):
            stats['dropped_placeholders'] += 1
            return None

    # Drop trailing empty key-value tails
    if EMPTY_TAIL_RE.search(line):
        stats['dropped_empty_tails'] += 1
        return None

    original = line
    line = line.strip()
    if not line:
        stats['empty_lines_removed'] += 1
        return None

    # Space fixes
    line = SPACE_BEFORE_PUNCT_RE.sub(r'\1', line)
    new_line = MULTISPACE_RE.sub(' ', line)
    if new_line != original.strip():
        stats['whitespace_collapsed'] += 1
    line = new_line

    # Date & float normalization
    line = normalize_date_in_line(line, stats)
    line = round_long_floats(line, stats)
    return line

def deduplicate_lines(lines: list[str], stats: dict) -> list[str]:
    seen = set()
    out = []
    for l in lines:
        if l in seen:
            stats['duplicates_removed'] += 1
            continue
        seen.add(l)
        out.append(l)
    return out

# ----------- Safety -----------
def assert_safety(input_dir: Path, output_dir: Path):
    raw_dir = Path("data/raw").resolve()
    if raw_dir == input_dir.resolve() or raw_dir in input_dir.resolve().parents:
        raise RuntimeError("Refusing to run: input_dir is inside data/raw/.")
    if input_dir.resolve() == output_dir.resolve():
        raise RuntimeError("Refusing to run: input_dir and output_dir are identical.")

# ----------- Processing -----------
def process_file(in_path: Path, out_path: Path, overwrite: bool) -> dict:
    stats = {
        'file': in_path.name,
        'status': 'ok',
        'original_lines': 0,
        'kept_lines': 0,
        'dropped_placeholders': 0,
        'dropped_empty_tails': 0,
        'empty_lines_removed': 0,
        'whitespace_collapsed': 0,
        'numbers_rounded': 0,
        'dates_normalized': 0,
        'duplicates_removed': 0,
    }

    if out_path.exists() and not overwrite:
        stats['status'] = 'skipped_exists'
        return stats

    try:
        with in_path.open('r', encoding='utf-8') as f:
            lines = f.readlines()
        stats['original_lines'] = len(lines)

        cleaned = []
        for line in lines:
            c = clean_line(line, stats)
            if c is not None:
                cleaned.append(c)

        cleaned = deduplicate_lines(cleaned, stats)
        stats['kept_lines'] = len(cleaned)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', encoding='utf-8', newline='\n') as f:
            f.write("\n".join(cleaned) + ("\n" if cleaned else ""))

    except Exception as e:
        stats['status'] = f'error:{e}'
    return stats

def write_report_with_summary(report_path: Path, rows: list[dict]):
    header = [
        'file','status','original_lines','kept_lines','dropped_placeholders',
        'dropped_empty_tails','empty_lines_removed','whitespace_collapsed',
        'numbers_rounded','dates_normalized','duplicates_removed'
    ]
    # Write rows
    with report_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in header})

        # Compute and append TOTAL row
        def _sum(key):
            return sum(int(r.get(key, 0) or 0) for r in rows if r.get('status','').startswith(('ok','skipped_exists','error')))
        total = {
            'file': 'TOTAL',
            'status': '',
            'original_lines': _sum('original_lines'),
            'kept_lines': _sum('kept_lines'),
            'dropped_placeholders': _sum('dropped_placeholders'),
            'dropped_empty_tails': _sum('dropped_empty_tails'),
            'empty_lines_removed': _sum('empty_lines_removed'),
            'whitespace_collapsed': _sum('whitespace_collapsed'),
            'numbers_rounded': _sum('numbers_rounded'),
            'dates_normalized': _sum('dates_normalized'),
            'duplicates_removed': _sum('duplicates_removed'),
        }
        w.writerow(total)

def main():
    # Defaults from config.yaml (project root)
    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_config(repo_root / "config.yaml")
    default_in, default_out = get_default_paths_from_config(cfg)

    parser = argparse.ArgumentParser(description="Clean processed EHR TXT files safely (with audit & summary).")
    parser.add_argument('--input_dir', type=str, default=str(default_in),
                        help='Directory containing parsed TXT files.')
    parser.add_argument('--output_dir', type=str, default=str(default_out),
                        help='Directory to write cleaned TXT files.')
    parser.add_argument('--overwrite', type=lambda x: str(x).lower() in {'1','true','yes'},
                        default=False, help='Overwrite existing cleaned files.')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"[ERROR] input_dir does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    assert_safety(input_dir, output_dir)

    txt_files = sorted([p for p in input_dir.glob("*.txt") if p.is_file()])
    if not txt_files:
        print(f"[WARN] No .txt files found in {input_dir}", file=sys.stderr)

    rows = []
    for p in txt_files:
        out_p = output_dir / p.name
        stats = process_file(p, out_p, overwrite=args.overwrite)
        rows.append(stats)
        tag = stats['status']
        print(f"[{tag:>12}] {p.name}  "
              f"(orig={stats['original_lines']} -> kept={stats['kept_lines']}, "
              f"drop={stats['dropped_placeholders'] + stats['dropped_empty_tails'] + stats['empty_lines_removed']}, "
              f"dups={stats['duplicates_removed']})")

    report_path = output_dir / "cleaning_report.csv"
    write_report_with_summary(report_path, rows)
    print(f"\n✔ Cleaning finished. Report (with TOTAL row): {report_path}")

if __name__ == "__main__":
    main()
# src/preprocessing/clean_ehr_text.py
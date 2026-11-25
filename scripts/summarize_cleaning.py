import os, re, csv, glob, time, json
from pathlib import Path
from statistics import median

def read_lines(dir_path):
    total_chars = 0
    lengths = []
    total_lines = 0
    for fp in glob.glob(str(Path(dir_path) / "*.txt")):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                s = line.rstrip("\n")
                total_lines += 1
                total_chars += len(s)
                lengths.append(len(s))
    return {
        "files": len(glob.glob(str(Path(dir_path) / "*.txt"))),
        "lines": total_lines,
        "chars": total_chars,
        "median_len": int(median(lengths) if lengths else 0),
    }

def summarize_audit(audit_dir_or_file):
    counters = {
        "placeholders_dropped": 0,
        "dates_normalized": 0,
        "decimals_rounded": 0,
        "empty_kv_removed": 0,
        "empty_tails_dropped": 0,
        "duplicates_removed": 0,
        "whitespace_collapsed": 0,
        "lines_modified": 0,
    }
    paths = []
    p = Path(audit_dir_or_file)
    if p.is_file():
        paths = [p]
    else:
        paths = list(p.glob("*.csv"))
    for fp in paths:
        with open(fp, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k in counters:
                    if k in row:
                        counters[k] += int(row[k] or 0)
    return counters

def main():
    before = read_lines("data/processed/parsed_txt")
    after = read_lines("data/processed/cleaned_txt")

    # If your cleaner logs a single audit.csv, point to it. Otherwise point to a folder of audits.
    audit = summarize_audit("data/processed/cleaned_txt/audit.csv")

    out = {
        "before": before,
        "after": after,
        "audit": audit
    }
    Path("reports").mkdir(exist_ok=True, parents=True)
    with open("reports/cleaning_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
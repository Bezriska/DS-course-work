"""Проверка, что вручную собранных объектов нет в исходных датасетах.

SMS Spam:
    python3 src/check_duplicates.py sms \
        --original data/sms_spam/raw/SMSSpamCollection \
        --new data/sms_spam/manual/messages.csv [--threshold 0.9]
  Ищет точные совпадения (после нормализации) и почти-дубликаты
  (похожесть по difflib >= threshold) — как с исходным датасетом, так и внутри новых данных.

Clotho:
    python3 src/check_duplicates.py clotho \
        --metadata-dir data/clotho/raw --new data/clotho/manual/sources.csv
  Сравнивает sound_id новых записей с sound_id из всех *metadata*.csv Clotho.
  Если в новом CSV не заполнен sound_id, он извлекается из ссылки вида .../sounds/<id>/.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import glob
import os
import re
import sys


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9£$€ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_sms_original(path: str) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n").split("\t", 1)[1] for line in f if "\t" in line]


def read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def check_sms(args: argparse.Namespace) -> int:
    original = read_sms_original(args.original)
    new_rows = read_csv(args.new)
    orig_norm = [normalize(t) for t in original]
    orig_set = set(orig_norm)
    problems = 0

    seen: dict[str, str] = {}
    for row in new_rows:
        rid, text = row.get("id", "?"), row.get("text", "")
        norm = normalize(text)
        if not norm:
            print(f"[пусто]       id={rid}")
            problems += 1
            continue
        if norm in orig_set:
            print(f"[в исходном]  id={rid}: {text[:80]!r}")
            problems += 1
        elif norm in seen:
            print(f"[повтор]      id={rid} = id={seen[norm]}: {text[:80]!r}")
            problems += 1
        else:
            best, best_text = 0.0, ""
            sm = difflib.SequenceMatcher(autojunk=False)
            sm.set_seq2(norm)
            for cand, cand_text in zip(orig_norm, original):
                sm.set_seq1(cand)
                if sm.real_quick_ratio() < args.threshold or sm.quick_ratio() < args.threshold:
                    continue
                r = sm.ratio()
                if r > best:
                    best, best_text = r, cand_text
            if best >= args.threshold:
                print(f"[похоже {best:.2f}] id={rid}: {text[:60]!r}\n               ~ {best_text[:60]!r}")
                problems += 1
        seen.setdefault(norm, rid)

    print(f"\nПроверено новых: {len(new_rows)}, исходных: {len(original)}, проблем: {problems}")
    return 1 if problems else 0


SOUND_ID_RE = re.compile(r"/sounds/(\d+)")


def extract_sound_id(row: dict) -> str | None:
    val = (row.get("sound_id") or "").strip()
    if val.isdigit():
        return val
    for key in ("url", "sound_link"):
        m = SOUND_ID_RE.search(row.get(key) or "")
        if m:
            return m.group(1)
    return None


def check_clotho(args: argparse.Namespace) -> int:
    files = glob.glob(os.path.join(args.metadata_dir, "**", "*metadata*.csv"), recursive=True)
    if not files:
        print(f"Не найдено *metadata*.csv в {args.metadata_dir}", file=sys.stderr)
        return 2
    clotho_ids: set[str] = set()
    for path in files:
        for row in read_csv(path):
            sid = extract_sound_id(row)
            if sid:
                clotho_ids.add(sid)
    print(f"sound_id в метаданных Clotho: {len(clotho_ids)} (файлов: {len(files)})")

    problems, seen = 0, set()
    new_rows = read_csv(args.new)
    for row in new_rows:
        rid, sid = row.get("id", "?"), extract_sound_id(row)
        if sid is None:
            print(f"[нет sound_id] id={rid}")
            problems += 1
        elif sid in clotho_ids:
            print(f"[в Clotho]     id={rid} sound_id={sid}")
            problems += 1
        elif sid in seen:
            print(f"[повтор]       id={rid} sound_id={sid}")
            problems += 1
        if sid:
            seen.add(sid)
    print(f"\nПроверено новых: {len(new_rows)}, проблем: {problems}")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="dataset", required=True)

    p_sms = sub.add_parser("sms")
    p_sms.add_argument("--original", required=True)
    p_sms.add_argument("--new", required=True)
    p_sms.add_argument("--threshold", type=float, default=0.9)

    p_cl = sub.add_parser("clotho")
    p_cl.add_argument("--metadata-dir", required=True)
    p_cl.add_argument("--new", required=True)

    args = parser.parse_args()
    return check_sms(args) if args.dataset == "sms" else check_clotho(args)


if __name__ == "__main__":
    sys.exit(main())

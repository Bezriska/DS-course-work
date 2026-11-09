"""Скачивание вручную выбранных записей Freesound и заполнение реестра источников.

Записи выбираются вручную на freesound.org: поиск сайта закрыт для роботов в robots.txt,
поэтому скрипт поиск не использует. Ссылки на страницы звуков записываются по одной
в строку в файл data/clotho/manual/urls.txt (строки с # игнорируются).

Для каждой ссылки скрипт:
  * берёт со страницы звука название, автора, лицензию, длительность, теги и ссылку на превью;
  * проверяет, что звука нет в Clotho (sound_id в метаданных из data/clotho/raw);
  * проверяет лицензию (CC0 / CC BY; CC BY-NC допускается с пометкой) и длительность 15–30 с;
  * скачивает HQ-превью (MP3 ~128 кбит/с) в data/clotho/manual/audio/fs_<sound_id>.mp3.
Оригиналы на Freesound доступны только после входа в аккаунт. Если они нужны,
их можно скачать вручную под тем же именем, но в .wav.

Результат — data/clotho/manual/sources.csv. Записи, не прошедшие проверки,
помечаются в столбце status и не скачиваются.

Пример:
    python3 src/fetch_freesound.py --collected-by maxim --prefix m
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import html
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANUAL = os.path.join(ROOT, "data", "clotho", "manual")
RAW = os.path.join(ROOT, "data", "clotho", "raw")
UA = "DS-course-work/0.1 (student coursework; manual collection of a few sounds)"

FIELDS = ["id", "file_name", "sound_id", "url", "title", "author", "license", "duration_sec",
          "tags", "access_date", "start_sec", "end_sec", "collected_by", "status", "notes"]

LICENSES = {
    "publicdomain/zero": ("CC0 1.0", True),
    "licenses/by/": ("CC BY", True),
    "licenses/by-nc/": ("CC BY-NC (только некоммерческое использование)", True),
    "licenses/sampling+": ("Sampling+", False),
}

BAD_TAGS = {"music", "song", "melody", "synth", "synthesizer", "loop", "beat", "drum-loop",
            "sound-design", "sfx", "game", "glitch", "voice-over", "speech", "podcast"}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def clotho_ids() -> set[str]:
    ids = set()
    for path in glob.glob(os.path.join(RAW, "*metadata*.csv")):
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("sound_id") or "").isdigit():
                    ids.add(row["sound_id"])
    return ids


def parse_page(page: str, url: str) -> dict:
    def first(pattern: str) -> str:
        m = re.search(pattern, page)
        return html.unescape(m.group(1)).strip() if m else ""

    lic_url = first(r"(creativecommons\.org/[a-z+/.\-0-9]+)")
    lic_name, lic_ok = next(((n, ok) for key, (n, ok) in LICENSES.items() if key in lic_url),
                            (lic_url or "не найдена", False))
    title = first(r'property="og:title" content="([^"]+)"')
    title = re.sub(r"\s+by\s+[^ ]+$", "", title)
    return {
        "sound_id": first(r'data-sound-id="(\d+)"') or first(r"/sounds/(\d+)"),
        "title": title,
        "author": first(r"/people/([^/]+)/sounds/") or "",
        "license": lic_name,
        "license_ok": lic_ok,
        "duration_sec": first(r'data-duration="([\d.]+)"'),
        "preview": first(r"(https://cdn\.freesound\.org/previews/[^\"']+-hq\.mp3)")
                   or first(r"(https://cdn\.freesound\.org/previews/[^\"']+\.mp3)"),
        "tags": sorted(set(re.findall(r"/browse/tags/([^/\"?]+)", page))),
        "url": url,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--urls", default=os.path.join(MANUAL, "urls.txt"))
    p.add_argument("--collected-by", default="maxim")
    p.add_argument("--prefix", default="m", help="префикс id, чтобы id участников не пересекались")
    p.add_argument("--min-dur", type=float, default=15.0)
    p.add_argument("--max-dur", type=float, default=30.0)
    p.add_argument("--delay", type=float, default=1.0, help="пауза между запросами, с")
    args = p.parse_args()

    with open(args.urls, encoding="utf-8") as f:
        urls = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    known = clotho_ids()
    if not known:
        print("Нет метаданных Clotho в data/clotho/raw: скачайте clotho_metadata_*.csv", file=sys.stderr)
        return 2

    audio_dir = os.path.join(MANUAL, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    today = dt.date.today().isoformat()
    rows, seen, n_ok = [], set(), 0

    for i, url in enumerate(urls, 1):
        row = {k: "" for k in FIELDS}
        row.update(id=f"{args.prefix}{i:03d}", url=url, access_date=today, collected_by=args.collected_by)
        try:
            info = parse_page(get(url).decode("utf-8", errors="replace"), url)
        except Exception as e:  # noqa: BLE001
            row["status"] = f"ошибка загрузки страницы: {e}"
            rows.append(row)
            continue
        sid = info["sound_id"]
        row.update(sound_id=sid, title=info["title"], author=info["author"], license=info["license"],
                   duration_sec=info["duration_sec"], tags=";".join(info["tags"]))
        dur = float(info["duration_sec"] or 0)
        bad_tags = BAD_TAGS & set(info["tags"])

        problems = []
        if sid in known:
            problems.append("уже есть в Clotho")
        if sid in seen:
            problems.append("повтор ссылки")
        if not info["license_ok"]:
            problems.append(f"лицензия не подходит ({info['license']})")
        if not (args.min_dur <= dur <= args.max_dur):
            problems.append(f"длительность {dur:.1f} с вне {args.min_dur:g}–{args.max_dur:g} с")
        if not info["preview"]:
            problems.append("нет ссылки на превью")
        seen.add(sid)

        if problems:
            row["status"] = "исключено: " + "; ".join(problems)
        else:
            fname = f"fs_{sid}.mp3"
            dest = os.path.join(audio_dir, fname)
            if not os.path.exists(dest):
                time.sleep(args.delay)
                with open(dest, "wb") as out:
                    out.write(get(info["preview"]))
            row.update(file_name=fname, start_sec="0", end_sec=f"{dur:.2f}", status="ok")
            if bad_tags:
                row["notes"] = "проверить на слух: теги " + ",".join(sorted(bad_tags))
            n_ok += 1
        rows.append(row)
        print(f"{row['id']} {sid:>7} {row['status'][:70]}")
        time.sleep(args.delay)

    out_path = os.path.join(MANUAL, "sources.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nГотово: {n_ok} из {len(urls)} записей подходят. Реестр: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

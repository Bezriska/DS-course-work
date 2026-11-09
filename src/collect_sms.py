"""Отбор реальных SMS для ручного обогащения SMS Spam Collection.

Источники (скачиваются в data/sms_spam/raw/, в git не хранятся):
  * spam — SMS Phishing Dataset (Mishra & Soni, Mendeley Data, 2022, CC BY 4.0),
           метки spam и smishing; сообщения распознаны со скриншотов из интернета;
  * ham  — NUS SMS Corpus (Chen & Kan, 2013), английская часть, выпуск 2015.03.09.

Фильтры: английский язык (без хинглиша и пиньиня), длина 20–300 символов и не меньше
4 слов, нет испорченных символов, в ham нет e-mail и длинных цифровых
последовательностей (персональные данные). Сообщения нет в исходном SMS Spam Collection:
ни точной копии, ни почти-копии, ни обрезанной версии (см. NearDupIndex
в check_duplicates.py). Оно также не повторяет уже отобранные.
Отбор случайный, seed фиксирован, поэтому результат воспроизводим.

Пример:
    python3 src/collect_sms.py --spam 50 --ham 50 --prefix m --collected-by maxim
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import random
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_duplicates import NearDupIndex, read_sms_original  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "sms_spam", "raw")
MANUAL = os.path.join(ROOT, "data", "sms_spam", "manual")

SOURCES = [
    {
        "source_id": "mendeley_smishing",
        "name": "SMS Phishing Dataset for Machine Learning and Pattern Recognition (Mishra & Soni, 2022)",
        "url": "https://data.mendeley.com/datasets/f45bkkt8pr/1",
        "license": "CC BY 4.0",
        "description": "Реальные spam/smishing SMS, распознанные со скриншотов из интернета; берём метки spam и smishing",
    },
    {
        "source_id": "nus_sms_corpus",
        "name": "NUS SMS Corpus, English, release 2015.03.09 (Chen & Kan, 2013)",
        "url": "https://github.com/kite1988/nus-sms-corpus",
        "license": "свободно для исследований с цитированием Chen & Kan (2013); отдельная лицензия в репозитории не указана",
        "description": "Реальные личные SMS, собранные у добровольцев NUS; берём как ham",
    },
]

EMAIL_RE = re.compile(r"\S+@\S+\.\w+")
LONG_DIGITS_RE = re.compile(r"\d[\d\s-]{6,}\d")

# Частые слова хинглиша (хинди латиницей) и пиньиня. Их нет в английском,
# поэтому два разных таких слова в сообщении — признак не-английского текста.
NON_ENGLISH_MARKERS = {
    # хинглиш
    "hai", "hain", "ka", "ki", "ke", "ko", "se", "mein", "aap", "aapka", "aapki", "apna", "apni",
    "karo", "karna", "nahi", "nahin", "bhi", "hoga", "wala", "wali", "sirf", "kal", "jaldi",
    "mujhe", "tum", "tera", "mera", "meri", "kya", "kyu", "kyun", "hum", "lete", "hue", "baje",
    "aaj", "abhi", "accha", "acha", "bhai", "yaar", "fayda", "ghar", "paisa", "kaise", "kab",
    # пиньинь
    "wo", "ni", "shi", "yi", "wei", "fei", "mian", "jiu", "zai", "hao", "bu", "zhe", "na", "ta",
    "men", "shen", "qu", "xiang", "dou", "mei", "gei", "zhi", "dao", "shuo", "xie",
}


def basic_ok(text: str) -> bool:
    text = text.strip()
    if not (20 <= len(text) <= 300) or len(text.split()) < 4:
        return False
    if "�" in text:  # символы, испорченные ещё в источнике
        return False
    if sum(ch.isascii() for ch in text) / len(text) <= 0.97:
        return False
    words = set(re.findall(r"[a-z]+", text.lower()))
    return len(words & NON_ENGLISH_MARKERS) < 2


def load_spam() -> list[dict]:
    path = os.path.join(RAW, "ext_mendeley", "Dataset_5971.csv")
    out = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):  # номер строки в CSV с учётом заголовка
            label = row["LABEL"].strip().lower()
            if label in ("spam", "smishing"):
                out.append({"text": row["TEXT"].strip(), "source_id": "mendeley_smishing",
                            "source_ref": f"Dataset_5971.csv:line{i}", "source_label": label})
    return out


def load_ham() -> list[dict]:
    path = os.path.join(RAW, "ext_nus", "smsCorpus_en_2015.03.09_all.xml")
    out = []
    for msg in ET.parse(path).getroot().iter("message"):
        prof = msg.find("messageProfile")
        if prof is not None and prof.get("language", "en") != "en":
            continue
        text = (msg.findtext("text") or "").strip()
        if EMAIL_RE.search(text) or LONG_DIGITS_RE.search(text):
            continue
        out.append({"text": text, "source_id": "nus_sms_corpus",
                    "source_ref": f"message id={msg.get('id')}", "source_label": "ham"})
    return out


def pick(pool: list[dict], k: int, checker: NearDupIndex, rng: random.Random) -> list[dict]:
    rng.shuffle(pool)
    chosen = []
    for item in pool:
        if len(chosen) == k:
            break
        if basic_ok(item["text"]) and checker.find(item["text"]) is None:
            checker.add(item["text"])
            chosen.append(item)
    if len(chosen) < k:
        raise SystemExit(f"Не хватило кандидатов: {len(chosen)} из {k}")
    return chosen


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spam", type=int, default=50)
    p.add_argument("--ham", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix", default="m", help="префикс id, чтобы id двух участников не пересекались")
    p.add_argument("--collected-by", default="maxim")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="CSV уже собранных сообщений (например, напарника), чтобы не взять их повторно")
    args = p.parse_args()

    rng = random.Random(args.seed)
    checker = NearDupIndex(read_sms_original(os.path.join(RAW, "SMSSpamCollection")))
    for path in args.exclude:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                checker.add(row["text"])

    spam = pick(load_spam(), args.spam, checker, rng)
    ham = pick(load_ham(), args.ham, checker, rng)
    items = spam + ham
    rng.shuffle(items)  # перемешиваем, чтобы порядок не подсказывал класс

    today = dt.date.today().isoformat()
    os.makedirs(MANUAL, exist_ok=True)
    out_path = os.path.join(MANUAL, "messages.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "source_id", "source_ref", "source_label",
                    "collected_by", "collected_at", "converted_from_email", "notes"])
        for i, it in enumerate(items, 1):
            w.writerow([f"{args.prefix}{i:03d}", it["text"], it["source_id"], it["source_ref"],
                        it["source_label"], args.collected_by, today, "no", ""])

    with open(os.path.join(MANUAL, "sources.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source_id", "name", "url", "access_date", "license", "description"])
        w.writeheader()
        for s in SOURCES:
            w.writerow({**s, "access_date": today})

    print(f"Сохранено {len(items)} сообщений ({len(spam)} spam, {len(ham)} ham) в {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

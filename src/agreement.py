"""Согласие двух аннотаторов по JSON-экспорту Label Studio.

Считает:
  * для полей-выборов (Choices): процент согласия, Cohen's kappa, матрицу совпадений;
  * для текстовых полей (TextArea): среднее сходство подписей двух аннотаторов
    (Jaccard по множествам слов, все пары «подпись A × подпись B»);
  * время разметки (lead_time) по каждому аннотатору.

Пример:
    python3 src/agreement.py labeling/annotations/sms_spam.json --choices label
    python3 src/agreement.py labeling/annotations/clotho.json \
        --choices main_source environment speech --texts caption_1 caption_2 \
        --out labeling/annotations/clotho_merged.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict


def annotator_id(annotation: dict) -> str:
    who = annotation.get("completed_by")
    if isinstance(who, dict):
        return str(who.get("email") or who.get("id"))
    return str(who)


def parse_export(path: str) -> tuple[dict, dict, dict]:
    """Возвращает (ответы, время, данные задач).

    ответы[annotator][task_id][field] -> str | list[str]
    время[annotator] -> list[float]
    """
    with open(path, encoding="utf-8") as f:
        tasks = json.load(f)

    answers: dict = defaultdict(dict)
    times: dict = defaultdict(list)
    task_data: dict = {}

    for task in tasks:
        task_id = task["id"]
        task_data[task_id] = task.get("data", {})
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            who = annotator_id(ann)
            fields: dict = {}
            for item in ann.get("result", []):
                name = item.get("from_name")
                value = item.get("value", {})
                if "choices" in value:
                    fields[name] = "|".join(sorted(value["choices"]))
                elif "text" in value:
                    fields[name] = " ".join(value["text"]).strip()
            answers[who][task_id] = fields
            if ann.get("lead_time") is not None:
                times[who].append(float(ann["lead_time"]))
    return answers, times, task_data


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    p_o = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if p_e == 1:
        return 1.0 if p_o == 1 else float("nan")
    return (p_o - p_e) / (1 - p_e)


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def jaccard(x: str, y: str) -> float:
    tx, ty = tokens(x), tokens(y)
    if not tx and not ty:
        return 1.0
    return len(tx & ty) / len(tx | ty)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", help="JSON-экспорт проекта Label Studio")
    parser.add_argument("--choices", nargs="*", default=[], help="имена полей Choices")
    parser.add_argument("--texts", nargs="*", default=[], help="имена полей TextArea с подписями")
    parser.add_argument("--annotators", nargs=2, metavar=("A", "B"),
                        help="id или email двух аннотаторов (по умолчанию первые два найденных)")
    parser.add_argument("--out", help="CSV с ответами обоих аннотаторов по каждой задаче")
    args = parser.parse_args()

    answers, times, task_data = parse_export(args.export)
    found = sorted(answers)
    if args.annotators:
        ann_a, ann_b = args.annotators
    elif len(found) >= 2:
        ann_a, ann_b = found[:2]
    else:
        print(f"Нужно минимум два аннотатора, найдено: {found}", file=sys.stderr)
        return 1
    if len(found) > 2 and not args.annotators:
        print(f"Внимание: аннотаторов больше двух ({found}), беру {ann_a} и {ann_b}\n")

    common = sorted(set(answers[ann_a]) & set(answers[ann_b]))
    print(f"Аннотатор A: {ann_a}, B: {ann_b}")
    print(f"Задач, размеченных обоими: {len(common)}\n")

    for field in args.choices:
        pairs = [(answers[ann_a][t].get(field), answers[ann_b][t].get(field)) for t in common]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        a = [x for x, _ in pairs]
        b = [y for _, y in pairs]
        agree = sum(x == y for x, y in pairs)
        print(f"== {field} (n={len(pairs)})")
        if pairs:
            print(f"  Процент согласия: {agree / len(pairs):.3f} ({agree}/{len(pairs)})")
            print(f"  Cohen's kappa:    {cohen_kappa(a, b):.3f}")
            print("  Матрица (A -> B):")
            for (x, y), cnt in sorted(Counter(pairs).items()):
                print(f"    {x:>12} -> {y:<12} {cnt}")
        print()

    if args.texts:
        sims = []
        for t in common:
            caps_a = [answers[ann_a][t].get(f, "") for f in args.texts]
            caps_b = [answers[ann_b][t].get(f, "") for f in args.texts]
            sims += [jaccard(x, y) for x in caps_a for y in caps_b if x and y]
        if sims:
            print(f"== Подписи {args.texts}: Jaccard A×B, среднее {statistics.mean(sims):.3f}, "
                  f"медиана {statistics.median(sims):.3f} (пар: {len(sims)})\n")

    print("== Время разметки (lead_time, сек)")
    for who in (ann_a, ann_b):
        tt = times.get(who, [])
        if tt:
            print(f"  {who}: n={len(tt)}, среднее {statistics.mean(tt):.1f}, "
                  f"медиана {statistics.median(tt):.1f}, всего {sum(tt) / 60:.1f} мин")

    if args.out:
        fields = args.choices + args.texts
        data_keys = sorted({k for t in common for k in task_data[t]})
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["task_id", *data_keys, *[f"{x}_a" for x in fields], *[f"{x}_b" for x in fields]])
            for t in common:
                w.writerow([t, *[task_data[t].get(k, "") for k in data_keys],
                            *[answers[ann_a][t].get(x, "") for x in fields],
                            *[answers[ann_b][t].get(x, "") for x in fields]])
        print(f"\nСохранено: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Согласие двух аннотаторов по JSON-экспортам Label Studio.

Два режима:
  * два файла — у каждого аннотатора своя установка Label Studio (наш случай):
        python3 src/agreement.py --a labeling/annotations/sms_maxim.json \
                                 --b labeling/annotations/sms_vitaliy.json --choices label
  * один файл — оба аннотатора работают в одном проекте:
        python3 src/agreement.py labeling/annotations/sms.json --choices label

Задачи из разных установок сопоставляются не по id Label Studio (он везде свой),
а по содержимому:
  * SMS — по полю msg_id;
  * аудио — по fs_<sound_id> в имени файла. Label Studio при загрузке добавляет
    к имени случайный префикс, поэтому сравниваем без него.

Считает:
  * для полей Choices — процент согласия, Cohen's kappa и матрицу совпадений;
  * для полей TextArea — среднее сходство подписей двух аннотаторов
    (Jaccard по множествам слов, все пары «подпись A × подпись B»);
  * время разметки (lead_time) у каждого аннотатора: среднее, медиану и сумму.
    Объекты дольше --max-time секунд (вкладку забыли открытой) не входят в среднее.

Пример для Clotho:
    python3 src/agreement.py --a labeling/annotations/clotho_maxim.json \
        --b labeling/annotations/clotho_vitaliy.json \
        --choices main_source environment speech --texts caption_1 caption_2 \
        --out labeling/annotations/clotho_merged.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

FS_RE = re.compile(r"fs_(\d+)")
UPLOAD_PREFIX_RE = re.compile(r"^[0-9a-f]{8}-")


def task_key(task: dict) -> str:
    data = task.get("data", {})
    if data.get("msg_id"):
        return str(data["msg_id"])
    for value in data.values():
        if isinstance(value, str):
            m = FS_RE.search(value)
            if m:
                return f"fs_{m.group(1)}"
    for value in data.values():
        if isinstance(value, str) and "/" in value:
            return UPLOAD_PREFIX_RE.sub("", os.path.basename(value))
    return f"task_{task.get('id')}"


def annotator_id(annotation: dict) -> str:
    who = annotation.get("completed_by")
    if isinstance(who, dict):
        return str(who.get("email") or who.get("id"))
    return str(who)


def parse_export(path: str) -> tuple[dict, dict, dict]:
    """Возвращает (ответы, время, данные задач).

    ответы[annotator][key][field] -> str
    время[annotator][key] -> float
    """
    with open(path, encoding="utf-8") as f:
        tasks = json.load(f)

    answers: dict = defaultdict(dict)
    times: dict = defaultdict(dict)
    task_data: dict = {}
    for task in tasks:
        key = task_key(task)
        task_data[key] = task.get("data", {})
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            who = annotator_id(ann)
            fields: dict = {}
            for item in ann.get("result", []):
                name, value = item.get("from_name"), item.get("value", {})
                if "choices" in value:
                    fields[name] = "|".join(sorted(value["choices"]))
                elif "text" in value:
                    fields[name] = " ".join(value["text"]).strip()
            answers[who][key] = fields
            if ann.get("lead_time") is not None:
                times[who][key] = float(ann["lead_time"])
    return answers, times, task_data


def single_annotator(path: str) -> tuple[dict, dict, dict]:
    answers, times, data = parse_export(path)
    if len(answers) != 1:
        print(f"Внимание: в {path} аннотаторов {len(answers)} ({sorted(answers)}), беру первого",
              file=sys.stderr)
    who = sorted(answers)[0]
    return answers[who], times.get(who, {}), data


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
    parser.add_argument("export", nargs="?", help="один JSON-экспорт, где есть оба аннотатора")
    parser.add_argument("--a", help="JSON-экспорт аннотатора A")
    parser.add_argument("--b", help="JSON-экспорт аннотатора B")
    parser.add_argument("--choices", nargs="*", default=[], help="имена полей Choices")
    parser.add_argument("--texts", nargs="*", default=[], help="имена полей TextArea с подписями")
    parser.add_argument("--annotators", nargs=2, metavar=("A", "B"),
                        help="для одного файла: id или email двух аннотаторов")
    parser.add_argument("--max-time", type=float, default=900,
                        help="объекты дольше стольких секунд не входят в среднее время")
    parser.add_argument("--out", help="CSV с ответами обоих аннотаторов по каждой задаче")
    args = parser.parse_args()

    if args.a and args.b:
        ans_a, time_a, data_a = single_annotator(args.a)
        ans_b, time_b, data_b = single_annotator(args.b)
        name_a = os.path.splitext(os.path.basename(args.a))[0]
        name_b = os.path.splitext(os.path.basename(args.b))[0]
        task_data = {**data_b, **data_a}
    elif args.export:
        answers, times, task_data = parse_export(args.export)
        found = sorted(answers)
        if args.annotators:
            name_a, name_b = args.annotators
        elif len(found) >= 2:
            name_a, name_b = found[:2]
        else:
            print(f"Нужно минимум два аннотатора, найдено: {found}. "
                  f"Если у каждого своя выгрузка, используйте --a и --b.", file=sys.stderr)
            return 1
        ans_a, ans_b = answers[name_a], answers[name_b]
        time_a, time_b = times.get(name_a, {}), times.get(name_b, {})
    else:
        parser.error("укажите один файл экспорта или пару --a/--b")

    common = sorted(set(ans_a) & set(ans_b))
    only_a, only_b = sorted(set(ans_a) - set(ans_b)), sorted(set(ans_b) - set(ans_a))
    print(f"Аннотатор A: {name_a} ({len(ans_a)} объектов), B: {name_b} ({len(ans_b)} объектов)")
    print(f"Размечено обоими: {len(common)}")
    if only_a or only_b:
        print(f"  только у A: {only_a[:10]}{' …' if len(only_a) > 10 else ''}")
        print(f"  только у B: {only_b[:10]}{' …' if len(only_b) > 10 else ''}")
    print()

    for field in args.choices:
        pairs = [(ans_a[k].get(field), ans_b[k].get(field)) for k in common]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        agree = sum(x == y for x, y in pairs)
        print(f"== {field} (n={len(pairs)})")
        if pairs:
            print(f"  Процент согласия: {agree / len(pairs):.3f} ({agree}/{len(pairs)})")
            print(f"  Cohen's kappa:    {cohen_kappa([x for x, _ in pairs], [y for _, y in pairs]):.3f}")
            print("  Матрица (A -> B):")
            for (x, y), cnt in sorted(Counter(pairs).items()):
                print(f"    {x:>12} -> {y:<12} {cnt}")
            diff = [k for k in common if ans_a[k].get(field) != ans_b[k].get(field)]
            if diff:
                print(f"  Расхождения: {', '.join(diff)}")
        print()

    if args.texts:
        sims = []
        for k in common:
            caps_a = [ans_a[k].get(f, "") for f in args.texts]
            caps_b = [ans_b[k].get(f, "") for f in args.texts]
            sims += [jaccard(x, y) for x in caps_a for y in caps_b if x and y]
        if sims:
            print(f"== Подписи {args.texts}: Jaccard A×B, среднее {statistics.mean(sims):.3f}, "
                  f"медиана {statistics.median(sims):.3f} (пар: {len(sims)})\n")

    print(f"== Время разметки (lead_time, сек; в среднее не входят объекты > {args.max_time:g} с)")
    for name, tt in ((name_a, time_a), (name_b, time_b)):
        vals = list(tt.values())
        if not vals:
            print(f"  {name}: нет данных")
            continue
        kept = [v for v in vals if v <= args.max_time]
        dropped = len(vals) - len(kept)
        mean = f"{statistics.mean(kept):.1f}" if kept else "—"
        med = f"{statistics.median(kept):.1f}" if kept else "—"
        print(f"  {name}: n={len(vals)}, среднее {mean}, медиана {med}, "
              f"всего {sum(kept) / 60:.1f} мин" + (f", отброшено выбросов: {dropped}" if dropped else ""))

    if args.out:
        fields = args.choices + args.texts
        data_keys = sorted({k for key in common for k in task_data.get(key, {})})
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["key", *data_keys, *[f"{x}_a" for x in fields], *[f"{x}_b" for x in fields],
                        "time_a", "time_b"])
            for key in common:
                w.writerow([key, *[task_data.get(key, {}).get(k, "") for k in data_keys],
                            *[ans_a[key].get(x, "") for x in fields],
                            *[ans_b[key].get(x, "") for x in fields],
                            time_a.get(key, ""), time_b.get(key, "")])
        print(f"\nСохранено: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

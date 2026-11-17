"""Итоговые метки SMS после разбора расхождений.

Объединяет:
  * собранные сообщения (data/sms_spam/manual/messages.csv);
  * разметку двух аннотаторов (JSON-экспорты Label Studio);
  * решения по расхождениям (labeling/annotations/sms_resolutions.csv).

Где аннотаторы совпали, итоговая метка равна их общей метке. Где разошлись,
она берётся из файла решений. Если для расхождения решения нет, скрипт
остановится с ошибкой.

Результат — data/sms_spam/manual/messages_labeled.csv со столбцами
label_a, label_b, unsure_a, unsure_b, label_final, resolution_rule (см. codebook, раздел 7).
В конце скрипт печатает, насколько итоговые метки совпадают с меткой источника
(smishing считается за spam).

    python3 src/finalize_labels.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agreement import single_annotator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--messages", default=p("data", "sms_spam", "manual", "messages.csv"))
    ap.add_argument("--a", default=p("labeling", "annotations", "sms_maxim.json"))
    ap.add_argument("--b", default=p("labeling", "annotations", "sms_vitaliy.json"))
    ap.add_argument("--resolutions", default=p("labeling", "annotations", "sms_resolutions.csv"))
    ap.add_argument("--out", default=p("data", "sms_spam", "manual", "messages_labeled.csv"))
    args = ap.parse_args()

    ans_a, _, _ = single_annotator(args.a)
    ans_b, _, _ = single_annotator(args.b)
    with open(args.resolutions, encoding="utf-8", newline="") as f:
        resolutions = {r["msg_id"]: r for r in csv.DictReader(f)}
    with open(args.messages, encoding="utf-8", newline="") as f:
        messages = list(csv.DictReader(f))

    rows, missing = [], []
    for m in messages:
        mid = m["id"]
        a, b = ans_a.get(mid, {}), ans_b.get(mid, {})
        la, lb = a.get("label", ""), b.get("label", "")
        if la and la == lb and mid not in resolutions:
            final, rule = la, "согласие"
        elif mid in resolutions:
            final, rule = resolutions[mid]["label_final"], resolutions[mid]["rule"]
        else:
            missing.append(mid)
            continue
        rows.append({**m, "label_a": la, "label_b": lb,
                     "unsure_a": "yes" if a.get("unsure") else "", "unsure_b": "yes" if b.get("unsure") else "",
                     "label_final": final, "resolution_rule": rule})
    if missing:
        print(f"Нет решения для расхождений: {', '.join(missing)}. Добавьте их в {args.resolutions}",
              file=sys.stderr)
        return 1

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    src = {"ham": "ham", "spam": "spam", "smishing": "spam"}
    diff = [r["id"] for r in rows if src.get(r["source_label"]) != r["label_final"]]
    n = len(rows)
    print(f"Итоговые метки: {n} сообщений, "
          f"ham {sum(r['label_final'] == 'ham' for r in rows)}, spam {sum(r['label_final'] == 'spam' for r in rows)}")
    print(f"Решено по файлу решений: {sum(r['resolution_rule'] != 'согласие' for r in rows)}")
    print(f"Совпадение с меткой источника: {n - len(diff)}/{n}" + (f", расходятся: {', '.join(diff)}" if diff else ""))
    print(f"Сохранено: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

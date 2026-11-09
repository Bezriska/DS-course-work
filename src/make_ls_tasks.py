"""Файл задач для импорта в Label Studio из собранных SMS.

В задачу попадают только id и текст. Метка источника (source_label) не попадает,
чтобы не подсказывать аннотаторам.

    python3 src/make_ls_tasks.py data/sms_spam/manual/messages.csv labeling/tasks/sms_spam_tasks.json
"""

import csv
import json
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8", newline="") as f:
        tasks = [{"data": {"msg_id": r["id"], "text": r["text"]}} for r in csv.DictReader(f)]
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=1)
    print(f"Задач: {len(tasks)} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

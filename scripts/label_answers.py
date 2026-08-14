"""Grade generated answers one at a time from the terminal.

Editing JSON by hand invites typos that silently become a third verdict class.
This shows one answer at a time, accepts a single keystroke, and writes after
every entry so grading can be interrupted and resumed.

    uv run python scripts/label_answers.py --run-id 86582a6b
"""

import argparse
import json
from pathlib import Path

JUDGEMENT_DIR = Path("data/eval/judgements")

KEYS = {
    "c": "correct",
    "p": "partial",
    "w": "wrong",
}

GUIDE = """
  c  correct   states the same fact as the reference
  p  partial   right but incomplete — e.g. a comparison answered for one card
  w  wrong     different figure, negated, or attached to the wrong thing
  s  skip      decide later
  q  quit      save and stop
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    path = JUDGEMENT_DIR / f"{args.run_id[:8]}_hand_labels.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path} — run make_label_sheet.py first")

    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]

    def save() -> None:
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    pending = [i for i, r in enumerate(records) if not r["verdict"]]
    print(f"\n  {len(records) - len(pending)}/{len(records)} already graded")
    print(GUIDE)

    for position, index in enumerate(pending, start=1):
        record = records[index]
        print("=" * 72)
        print(f"  {position} of {len(pending)}   {record['question_uid']}")
        print("=" * 72)
        print(f"\n  QUESTION\n    {record['question']}")
        print(f"\n  REFERENCE\n    {record['reference_answer']}")
        print(f"\n  SYSTEM ANSWER\n    {record['answer']}\n")

        while True:
            choice = input("  verdict [c/p/w/s/q] > ").strip().lower()
            if choice in KEYS:
                record["verdict"] = KEYS[choice]
                save()
                break
            if choice == "s":
                break
            if choice == "q":
                save()
                remaining = sum(1 for r in records if not r["verdict"])
                print(f"\n  saved. {remaining} still ungraded.\n")
                return
            print("    enter c, p, w, s or q")

    save()
    remaining = sum(1 for r in records if not r["verdict"])
    counts = {v: sum(1 for r in records if r["verdict"] == v) for v in KEYS.values()}
    print(f"\n  saved to {path}")
    print(f"  {counts}   ungraded {remaining}\n")


if __name__ == "__main__":
    main()

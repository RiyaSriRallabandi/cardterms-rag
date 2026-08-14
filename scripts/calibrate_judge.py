"""Compare judge verdicts against hand labels.

Reports Cohen's kappa alongside raw agreement. On a skewed set raw agreement is
misleading — a judge answering "correct" every time scores well and knows
nothing — so the corrected figure is the one that decides whether the judge's
verdicts can be reported.

    uv run python scripts/calibrate_judge.py --run-id 86582a6b
"""

import argparse
import json
from pathlib import Path

from cardterms.eval.stats import cohen_kappa

JUDGEMENT_DIR = Path("data/eval/judgements")

# Landis and Koch's conventional bands, quoted so the number is interpreted
# rather than merely reported.
BANDS = (
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.01, "slight"),
    (-1.0, "none"),
)


def load(path: Path) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    verdicts = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        verdicts[record["question_uid"]] = record["verdict"].strip().lower()
    return verdicts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompt", default="judge_v1")
    args = parser.parse_args()

    prefix = args.run_id[:8]
    judge = load(JUDGEMENT_DIR / f"{prefix}_{args.prompt}.jsonl")
    human = load(JUDGEMENT_DIR / f"{prefix}_hand_labels.jsonl")

    blank = [uid for uid, verdict in human.items() if not verdict]
    if blank:
        raise SystemExit(f"{len(blank)} hand labels are still empty, first: {blank[0]}")

    shared = sorted(set(judge) & set(human))
    if not shared:
        raise SystemExit("no questions graded by both")

    human_labels = [human[uid] for uid in shared]
    judge_labels = [judge[uid] for uid in shared]
    result = cohen_kappa(human_labels, judge_labels)

    band = next(name for threshold, name in BANDS if result["kappa"] >= threshold)

    print(f"\n  judge vs hand labels, n = {result['n']}")
    print(f"    raw agreement   {result['agreement']:.3f}")
    print(f"    Cohen's kappa   {result['kappa']:.3f}   ({band})\n")

    print("    human -> judge")
    for pair, count in sorted(result["confusion"].items()):
        if count:
            print(f"      {pair:24s} {count}")

    print("\n    disagreements")
    for uid in shared:
        if human[uid] != judge[uid]:
            print(f"      {uid:38s} human {human[uid]:8s} judge {judge[uid]}")

    # What the judge would report over everything it graded, stated separately
    # from how far it can be trusted.
    counts = {v: list(judge.values()).count(v) for v in ("correct", "partial", "wrong")}
    total = sum(counts.values())
    if total:
        print(f"\n    judge verdicts over all {total} answered questions")
        for verdict, count in counts.items():
            print(f"      {verdict:10s} {count:3d}   {count / total:.3f}")


if __name__ == "__main__":
    main()

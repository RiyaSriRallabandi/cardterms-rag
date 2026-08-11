"""Paired comparison of two evaluation runs.

Two tests are reported because they measure different things. McNemar's test
asks whether a relevant chunk reached the top k, which is binary and ignores a
passage moving from rank nine to rank six. The signed-rank test on reciprocal
rank uses the magnitude of every per-question change, and is the appropriate
test when a reranker improves ordering without crossing a threshold.

    uv run python scripts/compare_runs.py BASELINE_bm25_fixed_512_ov0 \
        bm25_fixed_512_ov0_rr-bge-50-aug
"""

import argparse

from cardterms.db import get_conn
from cardterms.eval.stats import mcnemar, wilcoxon_paired


def outcomes(conn, run_name: str) -> dict[int, dict]:
    rows = conn.execute(
        """
        SELECT rr.question_id,
               (rr.metrics->>'hit_rate@5')::float AS hit,
               (rr.metrics->>'mrr')::float AS mrr
        FROM run_results rr JOIN runs r ON r.id = rr.run_id
        WHERE r.run_name = %s
        """,
        (run_name,),
    ).fetchall()
    if not rows:
        raise SystemExit(f"no results for run {run_name!r}")
    return {r["question_id"]: {"hit": r["hit"] > 0, "mrr": r["mrr"]} for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("variant")
    args = parser.parse_args()

    with get_conn() as conn:
        base = outcomes(conn, args.baseline)
        var = outcomes(conn, args.variant)
        categories = {
            row["id"]: row["category"]
            for row in conn.execute(
                "SELECT id, category FROM eval_questions"
            ).fetchall()
        }

    shared = sorted(set(base) & set(var))
    b = [base[q]["hit"] for q in shared]
    v = [var[q]["hit"] for q in shared]

    print(f"{args.baseline}\n  vs\n{args.variant}\n")
    print(f"  n = {len(shared)}")

    hits = mcnemar(b, v)
    print(f"\n  hit_rate@5  {sum(b) / len(b):.3f} -> {sum(v) / len(v):.3f}")
    print(
        f"  fixed {hits['fixed']}   broken {hits['broken']}   "
        f"p = {hits['p_value']:.4f}  (McNemar)"
    )

    ranks = wilcoxon_paired(
        [base[q]["mrr"] for q in shared], [var[q]["mrr"] for q in shared]
    )
    mrr_base = sum(base[q]["mrr"] for q in shared) / len(shared)
    mrr_var = sum(var[q]["mrr"] for q in shared) / len(shared)
    print(f"\n  MRR         {mrr_base:.3f} -> {mrr_var:.3f}")
    print(
        f"  improved {ranks['improved']}   worsened {ranks['worsened']}   "
        f"p = {ranks['p_value']:.4f}  (Wilcoxon)"
    )

    print("\n  by category (hit_rate@5):")
    for category in sorted(set(categories.values())):
        ids = [q for q in shared if categories[q] == category]
        if not ids:
            continue
        bb = [base[q]["hit"] for q in ids]
        vv = [var[q]["hit"] for q in ids]
        print(
            f"    {category:20s} {sum(bb) / len(bb):.3f} -> {sum(vv) / len(vv):.3f}"
            f"   n={len(ids)}"
        )


if __name__ == "__main__":
    main()

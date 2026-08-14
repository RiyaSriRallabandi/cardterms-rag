"""Paired comparison of two evaluation runs.

Three tests are reported because they measure different things. McNemar's test
asks whether a relevant chunk reached the top k, which is binary and ignores a
passage moving from rank nine to rank six. The signed-rank test on reciprocal
rank uses the magnitude of every per-question change, and is the appropriate
test when a reranker improves ordering without crossing a threshold. The same
test on evidence coverage asks whether more of a question's documents reached
the context — the only one of the three that can see a comparison question go
from half-answered to fully answered.

    uv run python scripts/compare_runs.py BASELINE_bm25_fixed_512_ov0 \
        bm25_fixed_512_ov0_rr-bge-50-aug

Runs may be named or given as an id prefix. Names repeat across runs, so an
ambiguous name is rejected rather than silently blended.
"""

import argparse

from cardterms.db import get_conn
from cardterms.eval.stats import mcnemar, wilcoxon_paired


def resolve(conn, reference: str) -> dict:
    """Find one run by id prefix or exact name, refusing ambiguity."""
    rows = conn.execute(
        """
        SELECT id, run_name, started_at, notes
        FROM runs
        WHERE id::text LIKE %s OR run_name = %s
        ORDER BY started_at
        """,
        (reference + "%", reference),
    ).fetchall()

    if not rows:
        raise SystemExit(f"no run matching {reference!r}")
    if len(rows) > 1:
        detail = "\n".join(
            f"    {r['id']}  {r['started_at']:%Y-%m-%d %H:%M}  {r['notes'] or ''}"
            for r in rows
        )
        raise SystemExit(
            f"{reference!r} matches {len(rows)} runs — pass an id prefix:\n{detail}"
        )
    return rows[0]


def outcomes(conn, run_id) -> dict[int, dict]:
    rows = conn.execute(
        """
        SELECT rr.question_id,
               (rr.metrics->>'hit_rate@5')::float AS hit,
               (rr.metrics->>'mrr')::float        AS mrr,
               (rr.metrics->>'evidence@5')::float AS evidence
        FROM run_results rr
        WHERE rr.run_id = %s
        """,
        (run_id,),
    ).fetchall()
    if not rows:
        raise SystemExit(f"no results for run {run_id}")
    return {
        r["question_id"]: {
            "hit": r["hit"] > 0,
            "mrr": r["mrr"],
            "evidence": r["evidence"],
        }
        for r in rows
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("variant")
    args = parser.parse_args()

    with get_conn() as conn:
        base_run = resolve(conn, args.baseline)
        var_run = resolve(conn, args.variant)
        base = outcomes(conn, base_run["id"])
        var = outcomes(conn, var_run["id"])
        categories = {
            row["id"]: row["category"]
            for row in conn.execute(
                "SELECT id, category FROM eval_questions"
            ).fetchall()
        }

    shared = sorted(set(base) & set(var))
    b = [base[q]["hit"] for q in shared]
    v = [var[q]["hit"] for q in shared]

    print(
        f"{base_run['run_name']}  ({str(base_run['id'])[:8]})\n  vs\n"
        f"{var_run['run_name']}  ({str(var_run['id'])[:8]})\n"
    )
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

    # Evidence coverage predates only the runs that recorded it; older runs
    # store no value and are reported as unavailable rather than as zero.
    scored = [
        q
        for q in shared
        if base[q]["evidence"] is not None and var[q]["evidence"] is not None
    ]
    if scored:
        cov = wilcoxon_paired(
            [base[q]["evidence"] for q in scored],
            [var[q]["evidence"] for q in scored],
        )
        cov_base = sum(base[q]["evidence"] for q in scored) / len(scored)
        cov_var = sum(var[q]["evidence"] for q in scored) / len(scored)
        print(f"\n  evidence@5  {cov_base:.3f} -> {cov_var:.3f}")
        print(
            f"  improved {cov['improved']}   worsened {cov['worsened']}   "
            f"p = {cov['p_value']:.4f}  (Wilcoxon)"
        )
    else:
        print("\n  evidence@5  not recorded in one of these runs")

    print("\n  by category (hit_rate@5, evidence@5):")
    for category in sorted(set(categories.values())):
        ids = [q for q in shared if categories[q] == category]
        if not ids:
            continue
        bb = [base[q]["hit"] for q in ids]
        vv = [var[q]["hit"] for q in ids]
        line = f"    {category:20s} {sum(bb) / len(bb):.3f} -> {sum(vv) / len(vv):.3f}"
        cids = [q for q in ids if q in scored]
        if cids:
            eb = sum(base[q]["evidence"] for q in cids) / len(cids)
            ev = sum(var[q]["evidence"] for q in cids) / len(cids)
            line += f"    {eb:.3f} -> {ev:.3f}"
        print(f"{line}   n={len(ids)}")


if __name__ == "__main__":
    main()

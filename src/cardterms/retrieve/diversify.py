"""Reserve context slots for each card a question names.

Reranking scores every passage against the whole question independently, so a
question naming two cards can return five passages about the better-matching
one. The generator then either refuses, or compares against a card it was never
shown — and a wrong figure is worse than no figure.

Two strategies are implemented. The blind cap limits any document to a fixed
share of the window regardless of the question; it was measured first and
rejected, because it takes slots from single-card questions to pay for
multi-card ones. Entity-aware selection reserves slots only for cards the
question actually names, so single-card questions are left exactly as the
reranker ordered them.
"""

from cardterms.retrieve.entities import Entity


def diversify(
    ranked: list[tuple[int, int, float]],
    window: int,
    max_per_doc: int,
) -> list[tuple[int, int, float]]:
    """Limit any one document to `max_per_doc` of the first `window` results.

    Retained for the ablation in the report. Displaced passages follow in score
    order rather than being discarded, so metrics at greater depths are
    unaffected.
    """
    if max_per_doc <= 0 or window <= 0:
        return ranked

    selected: list[tuple[int, int, float]] = []
    deferred: list[tuple[int, int, float]] = []
    counts: dict[int, int] = {}

    for item in ranked:
        doc_id = item[1]
        if len(selected) < window and counts.get(doc_id, 0) < max_per_doc:
            selected.append(item)
            counts[doc_id] = counts.get(doc_id, 0) + 1
        else:
            deferred.append(item)

    return selected + deferred


def select_by_entity(
    ranked: list[tuple[int, int, float]],
    window: int,
    entities: list[Entity],
) -> list[tuple[int, int, float]]:
    """Guarantee each named card a share of the window, then fill by score.

    With fewer than two cards named there is nothing to balance and the
    reranker's order is returned untouched — this is what keeps single-card
    categories from paying for the fix.
    """
    if len(entities) < 2 or window <= 0:
        return ranked

    taken: set[int] = set()
    seen_docs: set[int] = set()
    selected: list[tuple[int, int, float]] = []

    def best(entity: Entity, fresh_doc: bool) -> tuple[int, int, float] | None:
        for item in ranked:
            chunk_id, doc_id, _ = item
            if chunk_id in taken or doc_id not in entity.doc_ids:
                continue
            if fresh_doc and doc_id in seen_docs:
                continue
            return item
        return None

    # Rotate across the named cards so no single one can take the window, and
    # within a card prefer a document not yet represented. A question naming
    # two variants of one brand needs a passage from each filing, not two
    # passages from whichever filing the reranker preferred.
    while len(selected) < window:
        progressed = False
        for entity in entities:
            if len(selected) >= window:
                break
            item = best(entity, fresh_doc=True) or best(entity, fresh_doc=False)
            if item is None:
                continue
            selected.append(item)
            taken.add(item[0])
            seen_docs.add(item[1])
            progressed = True
        if not progressed:
            break

    # Remaining slots, and everything below the window, in score order.
    remainder = [item for item in ranked if item[0] not in taken]
    return selected + remainder

"""Extend labels to other filings of the same agreement.

Several issuers file one agreement under multiple brand names; every such
filing answers a question about that agreement. Filings for different products
do not, even when they share wording, because retrieving the wrong product is
the error the evaluation exists to detect.

Document similarity cannot separate the two cases in this corpus: one issuer's
template is shared closely enough that different products score higher than
some genuine duplicates. The equivalence groups are therefore listed
explicitly.

    uv run python scripts/resolve_additional_labels.py           # report
    uv run python scripts/resolve_additional_labels.py --apply   # write
"""

import argparse
import json
from pathlib import Path

from cardterms.db import get_conn

GOLDEN_PATH = Path("data/eval/golden_set.jsonl")

# Filings of one agreement. Any member answers a question about any other.
EQUIVALENT_FILINGS = [
    # One agreement explicitly covering three brands.
    [
        "comenity_bank__bealls_inc_credit_card_bealls_florida_current_credit_card_agreements__255571",
        "comenity_bank__bealls_inc_credit_card_bealls_outlet_current_credit_card_agreements__255570",
        "comenity_bank__bealls_inc_credit_card_burkes_outlet_current_credit_card_agreements__256843",
    ],
    # Same agreement filed under two brand pages.
    [
        "comenity_capital_bank__saks_credit_card_current_credit_card_agreements__472174",
        "comenity_capital_bank__saks_world_elite_mastercard_credit_card_or_saks_credit_card_current_credit_card_agreements__472175",
    ],
    [
        "comenity_capital_bank__myacademy_rewards_credit_card_current_credit_card_agreements__593064",
        "comenity_capital_bank__myacademy_rewards_mastercard_credit_card_or_myacademy_rewards_credit_card_current_credit_card_agreements__593065",
    ],
    # Same filing submitted twice.
    [
        "onpoint_community_credit_union__consumer_variable_rate_cc_agreement_revised_11_4_21__2043",
        "onpoint_community_credit_union__consumer_variable_rate_cc_agreement_revised_11_4_21_final__255669",
    ],
    # One agreement, branded for an affinity programme.
    [
        "oregon_community_credit_union__occu_credit_card_account_agreement__256503",
        "oregon_community_credit_union__beaver_card_occu_credit_card_account_agreement_6971__537949",
    ],
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    group_of = {doc_uid: group for group in EQUIVALENT_FILINGS for doc_uid in group}

    entries = [
        json.loads(line)
        for line in GOLDEN_PATH.read_text().splitlines()
        if line.strip()
    ]

    with get_conn() as conn:
        documents = {
            row["doc_uid"]: row["raw_text"]
            for row in conn.execute(
                "SELECT doc_uid, raw_text FROM documents"
            ).fetchall()
        }

    added = 0
    missing = 0

    for entry in entries:
        existing = {label["doc_uid"] for label in entry.get("labels", [])}
        new_labels = []

        for label in list(entry.get("labels", [])):
            for sibling in group_of.get(label["doc_uid"], []):
                if sibling in existing:
                    continue
                if label["quote"] not in documents.get(sibling, ""):
                    missing += 1
                    print(
                        f"  quote absent from {sibling[:60]} ({entry['question_uid']})"
                    )
                    continue
                new_labels.append({**label, "doc_uid": sibling})
                existing.add(sibling)
                added += 1
                print(f"+ {entry['question_uid']}\n    {sibling}")

        entry.setdefault("labels", []).extend(new_labels)

    print(f"\n{added} label(s) added across {len(EQUIVALENT_FILINGS)} groups.")
    if missing:
        print(f"{missing} sibling(s) skipped: quote not present verbatim.")

    if args.apply:
        GOLDEN_PATH.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")
        print(f"Written to {GOLDEN_PATH}")
    else:
        print("Dry run. Re-run with --apply to write.")


if __name__ == "__main__":
    main()

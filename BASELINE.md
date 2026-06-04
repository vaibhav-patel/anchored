# BASELINE.md

> The measured starting line for **anchored**. Naive dense retrieval over CUAD. These numbers exist to be beaten — every later fix is justified by moving them.

_Generated 2026-06-04 09:25 UTC_

## Headline metrics

| Metric | Value |
|---|---|
| **recall@5** | **0.678** |
| **recall@10** | **0.8146** |
| **precision@5** | **0.1883** |
| cases evaluated | 205 |

## Configuration (what produced these numbers)

| Knob | Value |
|---|---|
| retriever | dense kNN (cosine), scoped to the target contract |
| embed model | `BAAI/bge-small-en-v1.5` |
| vector store | Elasticsearch (`dense_vector`) |
| chunk size / overlap | 512 / 64 tokens |
| chunker | tiktoken `cl100k_base`, fixed-size sliding window |
| index | `anchored_cuad` |

## Eval set

- **205 labeled cases** across **41 clause categories**, built from CUAD's expert annotations (`evals/cuad_retrieval.jsonl`).
- Gold spans relocated into normalized contract text by exact match; **0** answerable questions dropped as unalignable.

## How to read these

- **Task = within-contract clause retrieval.** CUAD's question text is a generic template (identical across all 510 contracts), so a corpus-wide query carries no signal about *which* contract to search. We therefore scope retrieval to the target contract — the realistic contract-review task (find clause X in *this* document). Pooling all contracts would measure title disambiguation, not clause retrieval.
- **recall@k** = a relevant chunk (overlapping a gold span) appears in the top-k. This is the metric that matters most for a retrieval baseline.
- **precision@5** has a low ceiling here: gold spans are sparse (often 1-2 relevant chunks per contract), so even perfect retrieval caps around 0.2-0.4. Treat it as a relative signal across experiments, not an absolute target.

> **Why scope to the contract (empirical):** the same eval run *without* the contract filter scores recall@5 = **0.175** (vs **0.678** scoped). Pooling all 510 contracts mostly measures whether the generic question text lands in the right *document* — not whether we retrieve the right *clause*. Scoping isolates the latter.

## Recall by clause category

| Category | n | recall@5 | recall@10 |
|---|---|---|---|
| Unlimited/All-You-Can-Eat-License | 5 | 0.0 | 0.2 |
| Volume Restriction | 5 | 0.4 | 0.4 |
| Most Favored Nation | 5 | 0.2 | 0.6 |
| Exclusivity | 5 | 0.4 | 0.6 |
| Non-Transferable License | 5 | 0.4 | 0.6 |
| Rofr/Rofo/Rofn | 5 | 0.6 | 0.6 |
| Affiliate License-Licensee | 5 | 0.6 | 0.6 |
| Liquidated Damages | 5 | 0.6 | 0.6 |
| Covenant Not To Sue | 5 | 0.2 | 0.8 |
| Affiliate License-Licensor | 5 | 0.2 | 0.8 |
| Irrevocable Or Perpetual License | 5 | 0.4 | 0.8 |
| Non-Disparagement | 5 | 0.4 | 0.8 |
| Document Name | 5 | 0.6 | 0.8 |
| Agreement Date | 5 | 0.6 | 0.8 |
| Renewal Term | 5 | 0.6 | 0.8 |
| No-Solicit Of Customers | 5 | 0.6 | 0.8 |
| No-Solicit Of Employees | 5 | 0.6 | 0.8 |
| License Grant | 5 | 0.6 | 0.8 |
| Parties | 5 | 0.8 | 0.8 |
| Governing Law | 5 | 0.8 | 0.8 |
| Audit Rights | 5 | 0.8 | 0.8 |
| Notice Period To Terminate Renewal | 5 | 0.8 | 0.8 |
| Termination For Convenience | 5 | 0.8 | 0.8 |
| Competitive Restriction Exception | 5 | 0.8 | 0.8 |
| Ip Ownership Assignment | 5 | 0.8 | 0.8 |
| Revenue/Profit Sharing | 5 | 0.8 | 0.8 |
| Source Code Escrow | 5 | 0.8 | 0.8 |
| Insurance | 5 | 0.6 | 1.0 |
| Non-Compete | 5 | 0.6 | 1.0 |
| Effective Date | 5 | 0.8 | 1.0 |
| Price Restrictions | 5 | 0.8 | 1.0 |
| Joint Ip Ownership | 5 | 0.8 | 1.0 |
| Expiration Date | 5 | 1.0 | 1.0 |
| Anti-Assignment | 5 | 1.0 | 1.0 |
| Minimum Commitment | 5 | 1.0 | 1.0 |
| Post-Termination Services | 5 | 1.0 | 1.0 |
| Warranty Duration | 5 | 1.0 | 1.0 |
| Change Of Control | 5 | 1.0 | 1.0 |
| Uncapped Liability | 5 | 1.0 | 1.0 |
| Cap On Liability | 5 | 1.0 | 1.0 |
| Third Party Beneficiary | 5 | 1.0 | 1.0 |

## Worst cases (error-analysis seeds for Phase 1)

| Category | top score | hit@5 | hit@10 |
|---|---|---|---|
| Document Name | 0.793 | ✗ | ✗ |
| Agreement Date | 0.818 | ✗ | ✗ |
| Governing Law | 0.813 | ✗ | ✗ |
| Exclusivity | 0.856 | ✗ | ✗ |
| Rofr/Rofo/Rofn | 0.853 | ✗ | ✗ |
| Parties | 0.830 | ✗ | ✗ |
| Exclusivity | 0.853 | ✗ | ✗ |
| Competitive Restriction Exception | 0.826 | ✗ | ✗ |
| No-Solicit Of Employees | 0.853 | ✗ | ✗ |
| License Grant | 0.831 | ✗ | ✗ |

## Reproduce

```bash
make data            # download + verify CUAD
make ingest && make index   # process → embed → index (~30 min, one-time)
make baseline        # regenerate this file
```

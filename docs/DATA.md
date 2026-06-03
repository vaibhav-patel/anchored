# CUAD — Data Documentation

The corpus for `anchored` is **CUAD v1** (Contract Understanding Atticus Dataset): 510
commercial contracts annotated by legal experts across 41 clause categories. It ships both
the raw contract text *and* labeled answer spans, which makes it a free, graded ground-truth
set for retrieval evaluation.

## Source & license

| | |
|---|---|
| Canonical archive | [Zenodo record 4595826](https://zenodo.org/records/4595826) (DOI-pinned) |
| Artifact | `CUAD_v1.zip` (~101 MB) |
| MD5 | `c38f490a984420b8a62600db401fafd5` |
| License | **CC BY 4.0** (attribution required) |
| Code / clause definitions | [TheAtticusProject/cuad](https://github.com/TheAtticusProject/cuad) |
| Project page | [atticusprojectai.org/cuad](https://www.atticusprojectai.org/cuad/) |

> **Attribution.** CUAD and the Atticus Labels are © The Atticus Project, released under
> CC BY 4.0. If you publish results, cite: *Hendrycks, Burns, Chen, Ball, "CUAD: An Expert-
> Annotated NLP Dataset for Legal Contract Review", NeurIPS 2021.*

## Acquisition

```bash
make data            # download → verify MD5 → extract → sanity counts
make data force=1    # force a re-download
```

`anchored/ingest/cuad.py` downloads the pinned Zenodo artifact, verifies the MD5 (re-fetches
on mismatch), extracts to `data/raw/CUAD_v1/`, and prints sanity counts. The step is
idempotent and `data/` is gitignored.

### Expected sanity counts

| Metric | Value |
|---|---|
| Contracts (`CUAD_v1.json` entries) | 510 |
| Contract text files (`full_contract_txt/*.txt`) | 510 |
| Clause categories | 41 |
| Total questions (510 × 41) | 20,910 |
| Answerable questions (≥1 labeled span) | 6,702 |

`make data` exits non-zero if any of these deviate from expectation.

## Extracted layout

```
data/raw/CUAD_v1/
├── CUAD_v1.json              # SQuAD-style annotations (the eval ground truth)
├── CUAD_v1_README.txt        # category descriptions + labeling notes
├── full_contract_txt/        # 510 plain-text contracts  ← ingestion input
├── full_contract_pdf/        # original PDFs (not used by the baseline)
└── label_group_xlsx/         # per-category spreadsheets
    master_clauses.csv        # flat clause table
```

For the Week 1 baseline we ingest **`full_contract_txt/`** (no PDF parsing needed) and use
**`CUAD_v1.json`** as the labeled retrieval ground truth.

## `CUAD_v1.json` schema (SQuAD-style)

```jsonc
{
  "version": "aok_v1.0",
  "data": [
    {
      "title": "LIMEENERGYCO_..._DISTRIBUTOR AGREEMENT",
      "paragraphs": [
        {
          "context": "<full contract text>",          // one paragraph per contract
          "qas": [
            {
              "id": "<title>__<Category>",              // e.g. ...__Document Name
              "question": "Highlight the parts ... Details: ...",
              "is_impossible": false,                    // true ⇒ category absent
              "answers": [
                { "text": "DISTRIBUTOR AGREEMENT",
                  "answer_start": 44 }                   // char offset into context
              ]
            }
            // ... 41 questions per contract (one per clause category)
          ]
        }
      ]
    }
    // ... 510 contracts
  ]
}
```

### Field notes that matter downstream

- **One paragraph per contract**: `data[i].paragraphs[0].context` is the entire contract.
  The same text appears (modulo whitespace) in `full_contract_txt/`.
- **`id` encodes the category**: split on `"__"` — the suffix is the clause category. This
  is how the eval harness (#5) groups questions by category.
- **`answer_start` is a character offset** into `context`. Combined with `len(text)`, it
  gives the exact gold span `[answer_start, answer_start + len(text))`. Chunk provenance in
  ingestion (#3) keeps `char_start`/`char_end` so a retrieved chunk can be checked for
  overlap against this gold span — the basis for recall@k / precision@k.
- **`is_impossible: true` / empty `answers`**: the category does not appear in that
  contract. ~14,208 of 20,910 questions are unanswerable; the 6,702 answerable ones are the
  usable retrieval ground truth.

## Caveats

- CUAD is **cleaner than a real corpus** (curated, single-document questions). We treat the
  raw text as messy on purpose (inconsistent headers, defined-term cross-references, long
  boilerplate) so the chunking failure modes induced in Phase 1 are real.
- The PDFs and xlsx are kept in `data/raw/` for completeness but are unused by the baseline.

"""Build notebooks/01_explore.ipynb from source.

Keeping the notebook's source as a plain script means it's reviewable in PRs and
regenerable, rather than an opaque JSON blob. Run this, then execute the notebook:

    python notebooks/build_explore.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_explore.ipynb

Both steps run inside the app container (where fastembed + Elasticsearch live).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


cells: list[nbf.NotebookNode] = []

# ───────────────────────────── Title ─────────────────────────────
cells.append(md(r"""
# anchored — Dataset & RAG Exploration

> A working notebook for the **anchored** project: explore the CUAD contract corpus, then
> probe the current naive dense-retrieval baseline with concrete **good** and **bad** cases.
>
> This is a build-in-public artifact — the charts and the failure walk-throughs are the point.

**Contents**
1. [The corpus](#1) — what CUAD is, in numbers and charts
2. [Chunks & spans](#2) — how the corpus is sliced for retrieval
3. [The RAG baseline](#3) — recall by category, score distributions
4. [✅ Good cases](#4) — queries that work, with the retrieved clause
5. [❌ Bad cases](#5) — failure modes, and *why* they fail
6. [Takeaways](#6) — what this points at for Phase 1
"""))

cells.append(code(r"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Run from the repo root regardless of where the kernel started (nbconvert uses the
# notebook's own directory as cwd).
if not Path("data").exists() and Path("../data").exists():
    os.chdir("..")

plt.rcParams["figure.figsize"] = (9, 4.5)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.size"] = 11

INK, OXBLOOD, SEAL, GOOD = "#1c1a17", "#7c2d2d", "#b8763c", "#4f8a4f"

DATA_DIR = Path("data")
CUAD_JSON = DATA_DIR / "raw" / "CUAD_v1" / "CUAD_v1.json"
print("cwd:", Path.cwd(), "| CUAD present:", CUAD_JSON.exists())
"""))

# ───────────────────────────── 1. Corpus ─────────────────────────────
cells.append(md(r"""
<a id="1"></a>
## 1. The corpus

[CUAD v1](https://www.atticusprojectai.org/cuad) (Contract Understanding Atticus Dataset,
CC BY 4.0): 510 commercial contracts, expert-annotated across 41 clause categories. Each
contract is asked all 41 questions; a question is *answerable* if that clause is present.
"""))

cells.append(code(r"""
with CUAD_JSON.open() as f:
    cuad = json.load(f)["data"]

rows = []
for entry in cuad:
    title = entry["title"]
    context = entry["paragraphs"][0]["context"]
    for qa in entry["paragraphs"][0]["qas"]:
        category = qa["id"].split("__", 1)[1] if "__" in qa["id"] else qa["id"]
        answers = qa.get("answers", [])
        rows.append({
            "contract": title,
            "category": category,
            "answerable": len(answers) > 0,
            "n_answers": len(answers),
            "contract_len": len(context),
            "answer_len": len(answers[0]["text"]) if answers else 0,
        })

df = pd.DataFrame(rows)
print(f"{df.contract.nunique()} contracts × {df.category.nunique()} categories "
      f"= {len(df):,} questions")
print(f"answerable: {df.answerable.sum():,}  ({df.answerable.mean():.1%})")
df.head()
"""))

cells.append(md("### Clause prevalence — how often each category actually appears"))

cells.append(code(r"""
prevalence = (
    df[df.answerable].groupby("category").size()
    .sort_values(ascending=False)
)
fig, ax = plt.subplots(figsize=(9, 9))
prevalence.sort_values().plot.barh(ax=ax, color=SEAL)
ax.set_title("Answerable questions per clause category (across 510 contracts)")
ax.set_xlabel("# contracts containing this clause")
ax.set_ylabel("")
plt.tight_layout(); plt.show()

print("Most common:", ", ".join(prevalence.head(3).index))
print("Rarest:     ", ", ".join(prevalence.tail(3).index))
"""))

cells.append(md(r"""
**Read:** the corpus is **imbalanced**. Some clauses (Parties, Governing Law, Document Name)
are in almost every contract; others (Most Favored Nation, Volume Restriction, the various
License sub-types) are rare. That long tail is exactly where retrieval will struggle.
"""))

cells.append(code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

contract_lens = df.groupby("contract").contract_len.first()
axes[0].hist(contract_lens / 1000, bins=40, color=OXBLOOD, alpha=0.85)
axes[0].set_title("Contract length")
axes[0].set_xlabel("thousands of characters"); axes[0].set_ylabel("# contracts")
axes[0].axvline(contract_lens.median() / 1000, color=INK, ls="--", lw=1,
                label=f"median {contract_lens.median()/1000:.0f}k")
axes[0].legend()

ans = df[df.answerable].answer_len
axes[1].hist(ans.clip(upper=2000), bins=40, color=SEAL, alpha=0.85)
axes[1].set_title("Gold answer-span length")
axes[1].set_xlabel("characters (clipped at 2000)"); axes[1].set_ylabel("# spans")
axes[1].axvline(ans.median(), color=INK, ls="--", lw=1, label=f"median {ans.median():.0f}")
axes[1].legend()

plt.tight_layout(); plt.show()
print(f"Contracts span {contract_lens.min()/1000:.0f}k–{contract_lens.max()/1000:.0f}k chars; "
      f"answer spans median {ans.median():.0f} chars.")
"""))

cells.append(md(r"""
**Read:** contracts are long (tens of thousands of characters) but gold answer spans are
**short** (median ~a few hundred chars). We're looking for a needle in each contract — which
is why chunking and ranking matter so much.
"""))

# ───────────────────────────── 2. Chunks ─────────────────────────────
cells.append(md(r"""
<a id="2"></a>
## 2. Chunks & spans

The pipeline slices each contract into overlapping token windows (tiktoken `cl100k_base`,
512 tokens / 64 overlap), keeping exact char offsets so a retrieved chunk can be checked
against a gold span.
"""))

cells.append(code(r"""
chunks_path = DATA_DIR / "processed" / "chunks.jsonl"
chunk_rows = []
if chunks_path.exists():
    with chunks_path.open() as f:
        for line in f:
            c = json.loads(line)
            chunk_rows.append({
                "contract_id": c["contract_id"],
                "chars": c["char_end"] - c["char_start"],
            })
    cdf = pd.DataFrame(chunk_rows)
    print(f"{len(cdf):,} chunks across {cdf.contract_id.nunique()} contracts "
          f"(avg {len(cdf)/cdf.contract_id.nunique():.1f} chunks/contract)")
    per = cdf.groupby("contract_id").size()
    fig, ax = plt.subplots()
    ax.hist(per, bins=40, color=OXBLOOD, alpha=0.85)
    ax.set_title("Chunks per contract"); ax.set_xlabel("# chunks"); ax.set_ylabel("# contracts")
    plt.tight_layout(); plt.show()
else:
    print("chunks.jsonl not found — run `make ingest` first.")
"""))

# ───────────────────────────── 3. Baseline ─────────────────────────────
cells.append(md(r"""
<a id="3"></a>
## 3. The RAG baseline

Now we run the actual retriever. We use the project's own eval harness over the live
Elasticsearch index (dense kNN, `bge-small-en-v1.5`), scoped to each target contract — the
realistic *find clause X in this document* task.
"""))

cells.append(code(r"""
from evals.dataset import build_cases, read_dataset, write_dataset
from evals.retrieval_eval import evaluate

ds_path = Path("evals/cuad_retrieval.jsonl")
if not ds_path.exists():
    cases, stats = build_cases("data", per_category=5)
    write_dataset(cases, ds_path)
    print("built", stats)
cases = read_dataset(ds_path)
print(f"{len(cases)} eval cases across {len({c.category for c in cases})} categories")

report = evaluate(cases)
print(f"recall@5={report.recall_at_5}  recall@10={report.recall_at_10}  "
      f"precision@5={report.precision_at_5}")
"""))

cells.append(md("### Recall by clause category — where retrieval wins and loses"))

cells.append(code(r"""
by_cat = report.by_category()
cat_df = (
    pd.DataFrame(by_cat).T[["recall@5", "recall@10"]]
    .sort_values(["recall@10", "recall@5"])
)
fig, ax = plt.subplots(figsize=(9, 10))
colors = [GOOD if v == 1.0 else (SEAL if v >= 0.6 else OXBLOOD) for v in cat_df["recall@10"]]
cat_df["recall@10"].plot.barh(ax=ax, color=colors)
ax.set_title("recall@10 by clause category (green=perfect, red=weak)")
ax.set_xlabel("recall@10"); ax.set_ylabel(""); ax.set_xlim(0, 1.05)
plt.tight_layout(); plt.show()

worst = cat_df.head(6); best = cat_df[cat_df["recall@10"] == 1.0]
print("WEAKEST:", ", ".join(worst.index))
print(f"PERFECT (recall@10=1.0): {len(best)} categories")
"""))

cells.append(md("### Score distributions — hits vs misses"))

cells.append(code(r"""
hit_scores = [c.top_score for c in report.case_results if c.hit_at_10 and c.top_score]
miss_scores = [c.top_score for c in report.case_results if not c.hit_at_10 and c.top_score]

fig, ax = plt.subplots()
ax.hist(hit_scores, bins=20, alpha=0.7, color=GOOD, label=f"hit@10 (n={len(hit_scores)})")
ax.hist(miss_scores, bins=20, alpha=0.7, color=OXBLOOD, label=f"miss@10 (n={len(miss_scores)})")
ax.set_title("Top-1 similarity score: hits vs misses")
ax.set_xlabel("cosine score of top result"); ax.set_ylabel("# cases"); ax.legend()
plt.tight_layout(); plt.show()

print(f"hits  top-score: mean {pd.Series(hit_scores).mean():.3f}")
print(f"misses top-score: mean {pd.Series(miss_scores).mean():.3f}")
"""))

cells.append(md(r"""
**Read:** misses don't have dramatically lower top scores — the model is often *confident
and wrong*. A raw score threshold alone won't separate them; we need lexical signal
(hybrid) and reranking, not just a cutoff.
"""))

# ───────────────────────────── 4. Good cases ─────────────────────────────
cells.append(md(r"""
<a id="4"></a>
## 4. ✅ Good cases

Let's look at retrieval *working*: pick cases that hit @5 and show the retrieved span next
to the gold clause. Helper below pretty-prints a case.
"""))

cells.append(code(r"""
from anchored.retrieve.search import dense_search

def show_case(case, k=5, max_chars=320):
    gold = case.gold_spans
    res = dense_search(case.question, k=k, contract_id=case.contract_id, trace=False)
    detail = case.question.split("Details:", 1)[-1].strip()
    print(f"CATEGORY : {case.category}")
    print(f"QUESTION : {detail[:130]}")
    print(f"CONTRACT : {case.contract_id[:70]}")
    print(f"GOLD SPAN: chars {gold[0][0]}–{gold[0][1]}")
    print("-" * 80)
    for i, r in enumerate(res, 1):
        rel = any(r.chunk.overlaps(s, e) for s, e in gold)
        mark = "✅" if rel else "  "
        snip = " ".join(r.chunk.text.split())[:max_chars]
        print(f"{mark} #{i} score={r.score:.3f} [{r.chunk.char_start}-{r.chunk.char_end}]")
        print(f"     {snip}…")
    print("=" * 80)

by_cat_cases = {}
for c in cases:
    by_cat_cases.setdefault(c.category, []).append(c)

# Pick strong categories with a clean hit.
good_targets = ["Cap On Liability", "Change Of Control", "Governing Law"]
for cat in good_targets:
    for c in by_cat_cases.get(cat, []):
        r = dense_search(c.question, k=5, contract_id=c.contract_id, trace=False)
        if any(any(x.chunk.overlaps(s, e) for s, e in c.gold_spans) for x in r):
            show_case(c); break
"""))

cells.append(md(r"""
**Read:** for short, lexically distinctive clauses the right span lands at rank 1 with a
high score and clearly overlaps the gold annotation. This is dense retrieval at its best.
"""))

# ───────────────────────────── 5. Bad cases ─────────────────────────────
cells.append(md(r"""
<a id="5"></a>
## 5. ❌ Bad cases — the failure modes

Now the interesting part. Two flavors of failure:

1. **Weak categories** — defined-term / diffuse clauses the embedder smooths away.
2. **Out-of-distribution queries** — questions CUAD has no answer to, where the system
   returns a confident-but-irrelevant span (no relevance gating).
"""))

cells.append(code(r"""
# 5a. Weak categories — show a miss.
bad_targets = ["Unlimited/All-You-Can-Eat-License", "Most Favored Nation", "Volume Restriction"]
for cat in bad_targets:
    for c in by_cat_cases.get(cat, []):
        r = dense_search(c.question, k=10, contract_id=c.contract_id, trace=False)
        if not any(any(x.chunk.overlaps(s, e) for s, e in c.gold_spans) for x in r[:5]):
            show_case(c, k=5); break
"""))

cells.append(md(r"""
**Read:** for these clauses the top results are *plausible-looking* contract prose at decent
scores, but none overlap the gold span — the relevant clause either wasn't chunked well or
ranks below the cutoff. No amount of staring at the score tells you it's wrong.
"""))

cells.append(code(r"""
# 5b. Out-of-distribution query — the "Connecticut" case from the demo.
ood = "I want discussions for Connecticut"
res = dense_search(ood, k=5, trace=False)  # NOT scoped — true open query
print(f"OOD QUERY: {ood!r}\n" + "-" * 80)
for i, r in enumerate(res, 1):
    snip = " ".join(r.chunk.text.split())[:200]
    print(f"  #{i} score={r.score:.3f}  {r.chunk.contract_id[:45]}")
    print(f"     {snip}…")
print("-" * 80)
print(f"top score {res[0].score:.3f} — high-looking, but the snippet is a table of "
      "store addresses. There is no 'discussions for Connecticut' clause in CUAD; the "
      "system has no way to say 'no confident match'.")
"""))

cells.append(md(r"""
**Read:** the OOD query returns a ~0.81 score — not obviously low — yet the content is
irrelevant (a data table). This is the canonical *confidently wrong* failure. A relevance
threshold / CRAG-style classifier is needed to abstain.
"""))

# ───────────────────────────── 6. Takeaways ─────────────────────────────
cells.append(md(r"""
<a id="6"></a>
## 6. Takeaways → Phase 1

| Observation (from this notebook) | Phase-1 fix |
|---|---|
| Weak on defined-term / rare clauses (License sub-types, MFN, Volume Restriction) | **Hybrid (BM25 + dense)** to recover exact-term signal |
| Misses often have high-ish scores; relevant chunk just outside top-k | **Reranking** over a larger candidate set |
| Long contracts, short gold spans; diffuse clauses chunked badly | **Contextual / late chunking** |
| OOD queries return confident-but-wrong spans | **Relevance threshold / CRAG** to abstain |

Each is a *break → measure → fix → write* loop, with this baseline as the starting line.
The numbers in this notebook are what we'll move.
"""))

# ───────────────────────────── Write ─────────────────────────────
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

out = Path(__file__).parent / "01_explore.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")

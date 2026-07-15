# Journal RAG + evaluation

## What this is

The agent can answer reflective/recall questions about the user's past journal
entries ("what have I been anxious about?", "summarize my mood this month") by
**retrieving** the most relevant entries and grounding its answer in them — a
retrieve → augment → generate (RAG) loop.

- **Embeddings:** OpenAI `text-embedding-3-small` (`app/embeddings.py`).
- **Vector store:** Pinecone serverless (managed) — `app/journal_index.py`.
- **Retrieval as a tool:** `search_journal` is a normal agent tool, so the
  existing supervisor loop *is* the RAG loop. The tool call is traced, so its
  output (the retrieved entries) is a span the eval can read.
- **Graceful degradation:** with no `PINECONE_API_KEY`, RAG no-ops — indexing
  is skipped and `search_journal` reports "unavailable". Everything else works.

## Setup

1. Create a free Pinecone account, get an API key, set `PINECONE_API_KEY`
   (locally and on Railway). Optionally set `PINECONE_INDEX_NAME`/`_CLOUD`/
   `_REGION` (defaults: `lifeops-journal` / `aws` / `us-east-1`). The index is
   created automatically on first use (dimension 1536, cosine).
2. New journal entries are indexed automatically on creation. To index entries
   created before RAG was enabled: `python scripts/backfill_journal_index.py`.
3. Try it: `journal: felt anxious about the demo today`, then
   `what have I been journaling about?` → the agent calls `search_journal` and
   answers from the retrieved entries.

## Evaluation — Langfuse managed evaluators (no local install)

We evaluate RAG quality with **Langfuse's managed LLM-as-judge evaluators**,
which run server-side on the app's traces — no eval library installed locally
(this is why we did NOT use the Ragas Python package: it's a heavy local-compute
dependency, and there's no thin-client hosted Ragas API). Prerequisite:
Langfuse auth must be working (valid `LANGFUSE_*` keys; the app was seen throwing
`401`, which must be fixed first, or no traces reach Langfuse).

Configure in the Langfuse dashboard (Evaluators → new LLM-as-judge evaluator),
mapping each evaluator's variables to the RAG trace:

| Evaluator | Measures (RAG triad) | Variable mapping |
|---|---|---|
| Hallucination / Faithfulness | Is the answer grounded in the retrieved entries? | `query` → trace input · `context` → `search_journal` observation output · `output` → trace output |
| Context Relevance | Are the retrieved entries relevant to the question? | `query` → trace input · `context` → `search_journal` observation output |
| Answer Relevance / Helpfulness | Does the answer address the question? | `query` → trace input · `output` → trace output |

Scope the evaluators to traces that contain a `search_journal` observation
(so they only score RAG turns). Scores then appear per-trace in Langfuse and
aggregate on the dashboard — the same faithfulness/context-relevance/
answer-relevance metrics Ragas computes, but hosted and continuous against real
traffic rather than a one-off script.

**Fallback if managed evaluators are gated on your Langfuse plan:** the RAG
triad can also be implemented as LLM-as-judge prompts using the existing
`evals/run_evals.py` `llm_judge()` helper (zero extra dependencies) — add RAG
cases with `judge_rubric`s asserting groundedness/relevance.

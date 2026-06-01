# Module 8 Lab — Vector Retrieval

Build the retrieval layer of a RAG system: ingest a technical-Q&A corpus into
Weaviate, implement BM25 / dense / hybrid retrievers, and evaluate all three
on a labeled set with recall@5, recall@10, and MRR.

Full assignment instructions are on the **Applied Lab page** in TalentLMS →
Module 8 → Applied Lab.

## Setup

1. Bring up Weaviate locally (image already pulled on EID-2):
   ```bash
   docker run -d --name weaviate-lab \
     -p 8080:8080 \
     -e DEFAULT_VECTORIZER_MODULE=none \
     -e ENABLE_MODULES= \
     -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
     semitechnologies/weaviate:1.24.10
   ```
   Confirm: `curl http://localhost:8080/v1/.well-known/ready` returns 200.

2. Install dependencies (cached from the drill — should be fast):
   ```bash
   pip install -r requirements.txt
   ```

3. Implement `retrieval.py` (the 6 functions: `create_schema`, `index_corpus`,
   `bm25_search`, `dense_search`, `hybrid_search`, `evaluate_retriever`).

4. Ingest the corpus once:
   ```bash
   python ingest.py
   ```
   Expect 1–3 minutes. Returns the count of ingested objects (~1,200).

5. Run the autograder locally:
   ```bash
   pytest tests/ -v
   ```

6. Fill in `comparison_brief.md` (~300–500 words).

7. Open a PR on branch `lab-8-vector-retrieval`. PR description must include
   the `evaluate_retriever` output for all three retrievers.

8. Paste your PR URL into TalentLMS → Module 8 → Applied Lab.

## Files

- `retrieval.py` — your implementation
- `ingest.py` — runnable driver (no edits needed)
- `comparison_brief.md` — your analysis (fill in)
- `setup-notes.md` — your troubleshooting record
- `data/corpus.jsonl` — the technical-Q&A corpus (~1,200 docs)
- `data/retrieval_eval.jsonl` — 60-pair labeled set (30 paraphrastic + 30 factoid)
- `tests/test_retrieval.py` — autograder
- `LICENSE` + `ATTRIBUTION.md` — corpus license (CC BY-SA) and attribution

## Resubmissions

Accepted through Saturday of the assignment week.

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms. Corpus content is governed separately by the CC BY-SA notices in [ATTRIBUTION.md](ATTRIBUTION.md) and `data/LICENSE`.

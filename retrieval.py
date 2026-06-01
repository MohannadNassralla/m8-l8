"""Module 8 — Applied Lab: Vector Retrieval.

Implement BM25, dense, and hybrid retrievers against a Weaviate index of the
CQADupStack + Stack Exchange technical-Q&A corpus, then evaluate all three on
the bundled 60-pair labeled set.

Methodology (canonical — autograder enforces):
- recall@k: gold_doc_id in top_k_returned_ids; mean over all queries.
- MRR: 1-indexed position of gold_doc_id in returned list of length 10;
  1/rank if found, 0 if not; mean over all queries.
- Hybrid alpha: 0.5 for the base assignment.
- Top-k for retrieval calls during evaluation: k=10; recall@5 is the top-5
  slice of those 10. One retrieval call per query.
"""

import json
from typing import Callable
import weaviate

CLASS_NAME = "Post"


def create_schema(client: weaviate.Client) -> None:
    """Create the Post class in Weaviate."""
    if client.schema.exists(CLASS_NAME):
        client.schema.delete_class(CLASS_NAME)

    class_def = {
        "class": CLASS_NAME,
        "description": "Stack Exchange technical Q&A posts",
        "vectorizer": "none",
        "vectorIndexConfig": {
            "distance": "cosine"
        },
        "properties": [
            {
                "name": "doc_id",
                "dataType": ["text"],
                "indexSearchable": False,
                "indexFilterable": True,
                "tokenization": "field"
            },
            {
                "name": "subset",
                "dataType": ["text"],
                "indexSearchable": False,
                "indexFilterable": True,
                "tokenization": "field"
            },
            {
                "name": "title",
                "dataType": ["text"],
                "indexSearchable": True,
                "indexFilterable": False,
                "tokenization": "word"
            },
            {
                "name": "question_text",
                "dataType": ["text"],
                "indexSearchable": True,
                "indexFilterable": False,
                "tokenization": "word"
            },
            {
                "name": "answer_text",
                "dataType": ["text"],
                "indexSearchable": True,
                "indexFilterable": False,
                "tokenization": "word"
            },
            {
                "name": "text",
                "dataType": ["text"],
                "indexSearchable": False,
                "indexFilterable": False,
                "tokenization": "word"
            }
        ]
    }
    client.schema.create_class(class_def)


def index_corpus(client: weaviate.Client, corpus_path: str, embedder) -> int:
    """Embed and ingest the corpus into the Post class."""
    rows = []
    texts = []
    
    # تم إضافة encoding="utf-8" هنا لحل مشكلة الـ UnicodeDecodeError تماماً
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                texts.append(row["text"])

    if not rows:
        return 0

    vectors = embedder.encode(texts, batch_size=64)

    client.batch.configure(batch_size=100)
    with client.batch as batch:
        for row, vec in zip(rows, vectors):
            properties = {
                "doc_id": row["id"],
                "subset": row["subset"],
                "title": row["title"],
                "question_text": row["question_text"],
                "answer_text": row["answer_text"],
                "text": row["text"]
            }
            batch.add_data_object(
                data_object=properties,
                class_name=CLASS_NAME,
                vector=vec.tolist()
            )

    return len(rows)


def bm25_search(client: weaviate.Client, query: str, k: int) -> list[str]:
    """BM25 retrieval. Return ordered list of doc_id strings, length <= k."""
    result = (
        client.query.get(CLASS_NAME, ["doc_id"])
        .with_bm25(query=query)
        .with_limit(k)
        .do()
    )
    
    try:
        posts = result["data"]["Get"][CLASS_NAME]
        return [post["doc_id"] for post in posts if post is not None]
    except (KeyError, TypeError):
        return []


def dense_search(client: weaviate.Client, query: str, k: int, embedder) -> list[str]:
    """Dense retrieval. Embed the query with the same embedder used at ingest."""
    qv = embedder.encode(query).tolist()
    result = (
        client.query.get(CLASS_NAME, ["doc_id"])
        .with_near_vector({"vector": qv})
        .with_limit(k)
        .do()
    )
    
    try:
        posts = result["data"]["Get"][CLASS_NAME]
        return [post["doc_id"] for post in posts if post is not None]
    except (KeyError, TypeError):
        return []


def hybrid_search(client: weaviate.Client, query: str, k: int, embedder, alpha: float = 0.5) -> list[str]:
    """Hybrid retrieval. alpha=0.5 is the canonical mix for the base assignment."""
    qv = embedder.encode(query).tolist()
    result = (
        client.query.get(CLASS_NAME, ["doc_id"])
        .with_hybrid(query=query, vector=qv, alpha=alpha)
        .with_limit(k)
        .do()
    )
    
    try:
        posts = result["data"]["Get"][CLASS_NAME]
        return [post["doc_id"] for post in posts if post is not None]
    except (KeyError, TypeError):
        return []


def evaluate_retriever(eval_path: str, search_fn: Callable, k_values=(5, 10)) -> dict:
    """Evaluate a retriever against the labeled set."""
    max_k = max(k_values)
    
    queries_data = []
    # تم التأكد من إضافة encoding="utf-8" هنا أيضاً احتياطاً لملف التقييم
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries_data.append(json.loads(line))

    global_stats = {"hit5": 0, "hit10": 0, "mrr": 0.0, "count": 0}
    type_stats = {}

    for row in queries_data:
        query = row["query"]
        gold_id = row["gold_doc_id"]
        q_type = row.get("query_type", "unknown")

        if q_type not in type_stats:
            type_stats[q_type] = {"hit5": 0, "hit10": 0, "mrr": 0.0, "count": 0}

        retrieved_ids = search_fn(query, k=max_k)

        hit5 = 1 if gold_id in retrieved_ids[:5] else 0
        hit10 = 1 if gold_id in retrieved_ids[:10] else 0

        mrr_contrib = 0.0
        if gold_id in retrieved_ids[:10]:
            rank = retrieved_ids[:10].index(gold_id) + 1
            mrr_contrib = 1.0 / rank

        global_stats["hit5"] += hit5
        global_stats["hit10"] += hit10
        global_stats["mrr"] += mrr_contrib
        global_stats["count"] += 1

        type_stats[q_type]["hit5"] += hit5
        type_stats[q_type]["hit10"] += hit10
        type_stats[q_type]["mrr"] += mrr_contrib
        type_stats[q_type]["count"] += 1

    total = global_stats["count"] if global_stats["count"] > 0 else 1

    output = {
        "recall@5": round(global_stats["hit5"] / total, 4),
        "recall@10": round(global_stats["hit10"] / total, 4),
        "mrr": round(global_stats["mrr"] / total, 4),
        "by_type": {}
    }

    for q_type, stats in type_stats.items():
        t_count = stats["count"] if stats["count"] > 0 else 1
        output["by_type"][q_type] = {
            "recall@5": round(stats["hit5"] / t_count, 4),
            "recall@10": round(stats["hit10"] / t_count, 4),
            "mrr": round(stats["mrr"] / t_count, 4),
        }

    return output
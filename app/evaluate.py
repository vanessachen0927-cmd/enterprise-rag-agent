import json
import math
import time
from pathlib import Path

from app.core import connect, retrieve

BASE = Path(__file__).resolve().parents[1]


class MissingBenchmarkSourcesError(ValueError):
    def __init__(self, sources):
        self.sources = sources
        super().__init__(
            'Load the sample policy dataset before evaluation. '
            f'Missing benchmark sources: {", ".join(sources)}'
        )


def evaluate_retrieval(
    dataset=None,
    top_k=3,
    db=None,
    retriever=None,
    mode='lexical',
    embedder=None,
):
    if mode not in {'lexical', 'semantic'}:
        raise ValueError('mode must be lexical or semantic')
    dataset = dataset or json.loads((BASE / 'data' / 'eval_questions.json').read_text())
    with connect(db) as con:
        available_sources = {
            row[0] for row in con.execute('SELECT DISTINCT source FROM chunks')
        }
    missing_sources = sorted(
        {case['source'] for case in dataset} - available_sources
    )
    if missing_sources:
        raise MissingBenchmarkSourcesError(missing_sources)
    if retriever is None and mode == 'semantic':
        from app.semantic import semantic_retrieve

        retriever = lambda question, count: semantic_retrieve(
            question,
            top_k=count,
            db=db,
            embedder=embedder,
        )
    elif retriever is None:
        retriever = lambda question, count: retrieve(question, top_k=count, db=db)
    per_case = []
    for case in dataset:
        started = time.perf_counter()
        hits = retriever(case['question'], top_k)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        success = any(hit['source'] == case['source'] for hit in hits)
        per_case.append({
            'question': case['question'],
            'expected': case['source'],
            'pass': success,
            'latency_ms': latency_ms,
            'retrieved': len(hits),
        })
    latencies = sorted(case['latency_ms'] for case in per_case)
    total = len(per_case)
    p95_index = max(0, math.ceil(0.95 * total) - 1)
    return {
        'mode': mode,
        'total': total,
        'hits': sum(case['pass'] for case in per_case),
        'recall_at_k': round(sum(case['pass'] for case in per_case) / total, 4) if total else 0,
        'mean_latency_ms': round(sum(latencies) / total, 3) if total else 0,
        'p95_latency_ms': latencies[p95_index] if latencies else 0,
        'cases': per_case,
    }


if __name__ == '__main__':
    from app.core import ingest, load_demo

    ingest(load_demo())
    print(json.dumps(evaluate_retrieval(top_k=1), indent=2))

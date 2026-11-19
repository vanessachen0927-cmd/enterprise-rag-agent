import pytest
from types import SimpleNamespace
from app.agent import plan, run_agent
from app.core import ingest, load_demo
from app.semantic import create_vector_index, semantic_retrieve, cosine
from app.evaluate import evaluate_retrieval, MissingBenchmarkSourcesError

@pytest.fixture
def db(tmp_path):
    db = tmp_path / 'semantic.db'
    ingest(load_demo(), db)
    return db

def fake_embed(texts):
    def v(text):
        text = text.lower()
        return [float('return' in text or 'refund' in text), float('inventory' in text or 'sage' in text), float('order' in text or 'carrier' in text), float('security' in text or 'api keys' in text)]
    return [v(t) for t in texts]

def test_build_and_retrieve_vectors(db):
    assert create_vector_index(db=db, embedder=fake_embed)['vectors'] == 4
    hits = semantic_retrieve('return refund', db=db, embedder=fake_embed)
    assert hits[0]['source'] == 'returns_policy.md'

def test_stale_index_rejected(db):
    create_vector_index(db=db, embedder=fake_embed)
    ingest([{'source': 'new', 'text': 'New content'}], db)
    with pytest.raises(ValueError, match='stale'):
        semantic_retrieve('hello', db=db, embedder=fake_embed)

def test_agent_trace_and_approval(db):
    result = run_agent('reorder SKU-22 available -2 threshold 8', db=db)
    assert result['requires_approval']
    assert result['answer']['reorder_units'] == 8
    assert result['trace'][0]['tool'] == 'inventory_reorder'

def test_agent_semantic(db):
    create_vector_index(db=db, embedder=fake_embed)
    result = run_agent('return refund', db=db, mode='semantic', embedder=fake_embed)
    assert result['citations'][0]['source'] == 'returns_policy.md'
    assert result['latency_ms'] >= 0

def test_semantic_and_ollama_fallback(db):
    class UnavailableOllama:
        def chat(self, **kwargs):
            raise ConnectionError('Ollama unavailable')

    result = run_agent(
        'When can a customer return an item?',
        db=db,
        mode='semantic',
        llm_client=UnavailableOllama(),
    )
    assert result['citations'][0]['source'] == 'returns_policy.md'
    assert result['fallback']['to'] == 'lexical'
    assert result['trace'][-2]['status'] == 'ollama_unavailable'

def test_ollama_generated_answer(db):
    class AvailableOllama:
        def chat(self, **kwargs):
            return SimpleNamespace(
                message=SimpleNamespace(content='Returns are allowed within 30 days.')
            )

    result = run_agent(
        'When can a customer return an item?',
        db=db,
        llm_client=AvailableOllama(),
    )
    assert result['answer'] == 'Returns are allowed within 30 days.'
    assert result['trace'][-2]['status'] == 'ollama'
    assert result['citations'][0]['source'] == 'returns_policy.md'

def test_evaluation(db):
    report = evaluate_retrieval(db=db, top_k=1)
    assert report['total'] == 8
    assert 0 <= report['recall_at_k'] <= 1
    assert report['p95_latency_ms'] >= 0

def test_evaluation_requires_benchmark_sources(tmp_path):
    db = tmp_path / 'custom-only.db'
    ingest([{'source': 'custom.md', 'text': 'Custom document.'}], db)
    with pytest.raises(MissingBenchmarkSourcesError, match='Load the sample policy dataset'):
        evaluate_retrieval(db=db, mode='lexical')

def test_zero_vectors():
    assert cosine([0,0],[2,3]) == 0

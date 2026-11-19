import pytest
from app.core import answer, chunks, ingest, inventory_reorder_tool, retrieve, load_demo, estimate_tokens

@pytest.fixture
def db(tmp_path):
    path = tmp_path / 'test.db'
    ingest(load_demo(), path)
    return path

def test_demo_ingest(db):
    assert len(retrieve('returns delivery packaging', db=db)) > 0

def test_citation(db):
    response = answer('returns receipt original order number', db=db)
    assert response['citations'][0]['source'] == 'returns_policy.md'

def test_grounded_extractive(db, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    response = answer('negative sage available quantities customer reports', db=db)
    assert response['mode'] == 'extractive_rag'
    assert response['citations'][0]['source'] == 'inventory_policy.md'

def test_openai_key_does_not_enable_paid_calls(db, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'not-used')
    response = answer('When can customers request returns?', db=db)
    assert response['mode'] == 'extractive_rag'
    assert response['citations'][0]['source'] == 'returns_policy.md'

def test_abstain(db):
    assert answer('xylophone nebula cryptogram', db=db)['citations'] == []

def test_agent_tool(db):
    response = answer('reorder SKU-12 available -4 threshold 10', db=db)
    assert response['answer']['reorder_units'] == 10
    assert response['mode'] == 'tool'

def test_tool_threshold():
    with pytest.raises(ValueError):
        inventory_reorder_tool('A', 1, -1)

def test_retrieval_top_k(db):
    assert len(retrieve('Sage Shopify order inventory', top_k=2, db=db)) <= 2

def test_chunk_overlap():
    out = chunks(' '.join(str(i) for i in range(12)), size=5, overlap=2)
    assert out[0].split()[-2:] == out[1].split()[:2]

def test_chunk_invalid():
    with pytest.raises(ValueError):
        chunks('hello world', size=2, overlap=2)

def test_context_budget(db):
    result = answer('refund original payment method', top_k=1, db=db)
    assert result['estimated_context_tokens'] > 0
    assert estimate_tokens('12345678') == 2

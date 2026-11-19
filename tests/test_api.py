from fastapi.testclient import TestClient
from app.api import app
from app.core import connect

client = TestClient(app)

def test_health():
    response = client.get('/health')
    assert response.status_code == 200

def test_api_demo_and_ask(monkeypatch, tmp_path):
    monkeypatch.setattr('app.core.DB', tmp_path / 'api.db')
    assert client.post('/demo/load').json()['chunks'] == 4
    response = client.post('/ask', json={'question':'When can customers request returns?'})
    assert response.status_code == 200
    assert response.json()['citations'][0]['source'] == 'returns_policy.md'

def test_api_validation():
    assert client.post('/ask', json={'question':''}).status_code == 422

def test_ingest_streamlit_payload(monkeypatch, tmp_path):
    monkeypatch.setattr('app.core.DB', tmp_path / 'ingest.db')
    payload = [{'source': 'policy.md', 'text': 'Returns are accepted within 30 days.'}]
    response = client.post('/ingest', json=payload)
    assert response.status_code == 200
    assert response.json()['chunks'] == 1

def test_demo_load_preserves_custom_documents(monkeypatch, tmp_path):
    database = tmp_path / 'demo.db'
    monkeypatch.setattr('app.core.DB', database)
    custom = [{'source': 'custom.md', 'text': 'A custom policy document.'}]
    assert client.post('/ingest', json=custom).status_code == 200
    assert client.post('/demo/load').status_code == 200
    assert client.post('/demo/load').status_code == 200
    with connect(database) as con:
        sources = {
            row[0] for row in con.execute('SELECT DISTINCT source FROM chunks')
        }
    assert sources == {
        'custom.md',
        'returns_policy.md',
        'inventory_policy.md',
        'order_policy.md',
        'security_policy.md',
    }

def test_semantic_index_uses_local_indexer(monkeypatch):
    monkeypatch.setattr(
        'app.semantic.create_vector_index',
        lambda: {'vectors': 4, 'model': 'local-test-model', 'dimensions': 4},
    )
    response = client.post('/semantic/index')
    assert response.status_code == 200
    assert response.json()['vectors'] == 4

def test_public_ingest_limit(monkeypatch):
    monkeypatch.setattr('app.api.PUBLIC_DEMO_MODE', True)
    monkeypatch.setattr('app.api.MAX_DOCUMENTS', 8)
    documents = [{'source': str(i), 'text': 'policy'} for i in range(9)]
    assert client.post('/ingest', json=documents).status_code == 413

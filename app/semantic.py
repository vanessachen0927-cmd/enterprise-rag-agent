import hashlib
import math
import os
from functools import lru_cache
from pathlib import Path

from app.config import CHROMA_DIR, EMBEDDING_MODEL, HF_CACHE_DIR
from app.core import DB, connect


def cosine(a, b):
    denominator = math.sqrt(sum(value * value for value in a)) * math.sqrt(
        sum(value * value for value in b)
    )
    return sum(left * right for left, right in zip(a, b)) / denominator if denominator else 0.0


def vector_store_path(db=None):
    database = Path(db or DB).resolve()
    if database == DB.resolve():
        return CHROMA_DIR
    return database.with_name(database.name + '.chroma')


def _collection_name(model):
    suffix = hashlib.sha256(model.encode('utf-8')).hexdigest()[:12]
    return f'policy_chunks_{suffix}'


def _collection(db=None, model=None):
    import chromadb

    model_name = model or EMBEDDING_MODEL
    path = vector_store_path(db)
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(
        name=_collection_name(model_name),
        metadata={'hnsw:space': 'cosine', 'model': model_name},
        embedding_function=None,
    )
    return collection


@lru_cache(maxsize=2)
def _embedding_model(model_name):
    from sentence_transformers import SentenceTransformer

    os.environ['HF_HOME'] = str(HF_CACHE_DIR)
    return SentenceTransformer(
        model_name,
        cache_folder=str(HF_CACHE_DIR),
        device='cpu',
    )


def embed_local(texts, model=None):
    if not texts:
        return []
    model_name = model or EMBEDDING_MODEL
    vectors = _embedding_model(model_name).encode(
        texts,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def _current_chunks(db):
    with connect(db) as source:
        rows = source.execute(
            'SELECT id, source, text FROM chunks ORDER BY id'
        ).fetchall()
    return {int(row[0]): (row[1], row[2]) for row in rows}


def semantic_index_status(db=None, model=None):
    model_name = model or EMBEDDING_MODEL
    path = vector_store_path(db)
    if not path.exists():
        return {'status': 'not_built', 'indexed_chunks': 0, 'model': model_name}
    import chromadb

    client = chromadb.PersistentClient(path=str(path))
    name = _collection_name(model_name)
    if name not in {collection.name for collection in client.list_collections()}:
        return {'status': 'not_built', 'indexed_chunks': 0, 'model': model_name}
    collection = client.get_collection(name=name, embedding_function=None)
    indexed = collection.get(include=['documents', 'metadatas'])
    indexed_chunks = len(indexed['ids'])
    current = _current_chunks(db)
    stored = {
        int(metadata['chunk_id']): (metadata['source'], document)
        for metadata, document in zip(indexed['metadatas'], indexed['documents'])
    }
    status = 'ready' if stored == current and indexed_chunks > 0 else 'stale'
    return {
        'status': status,
        'indexed_chunks': indexed_chunks,
        'model': model_name,
    }


def create_vector_index(db=None, embedder=None, model=None):
    model_name = model or EMBEDDING_MODEL
    current = _current_chunks(db)
    collection = _collection(db, model_name)
    existing = collection.get(include=['metadatas'])
    if existing['ids']:
        collection.delete(ids=existing['ids'])
    if not current:
        return {'vectors': 0, 'model': model_name, 'dimensions': 0}
    texts = [text for _, text in current.values()]
    vectors = (embedder or (lambda items: embed_local(items, model_name)))(texts)
    if len(vectors) != len(current) or any(not vector for vector in vectors):
        raise ValueError('Embedding count or dimensions invalid')
    dimensions = len(vectors[0])
    if any(len(vector) != dimensions for vector in vectors):
        raise ValueError('Inconsistent embedding dimensions')
    ids = [str(chunk_id) for chunk_id in current]
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=[
            {'chunk_id': chunk_id, 'source': source}
            for chunk_id, (source, _) in current.items()
        ],
        embeddings=[list(map(float, vector)) for vector in vectors],
    )
    return {'vectors': len(vectors), 'model': model_name, 'dimensions': dimensions}


def semantic_retrieve(query, top_k=3, db=None, embedder=None, model=None):
    if top_k < 1:
        raise ValueError('top_k must be positive')
    model_name = model or EMBEDDING_MODEL
    collection = _collection(db, model_name)
    if collection.count() == 0:
        raise ValueError('No semantic index. Run /semantic/index first.')
    current = _current_chunks(db)
    stored = collection.get(include=['documents', 'metadatas'])
    indexed = {
        int(metadata['chunk_id']): (metadata['source'], document)
        for metadata, document in zip(stored['metadatas'], stored['documents'])
    }
    if indexed != current:
        raise ValueError('Semantic index is stale. Run /semantic/index again.')
    query_vector = (embedder or (lambda items: embed_local(items, model_name)))([query])[0]
    results = collection.query(
        query_embeddings=[list(map(float, query_vector))],
        n_results=min(top_k, len(indexed)),
        include=['documents', 'metadatas', 'distances'],
    )
    hits = []
    for chunk_id, source, text, distance in zip(
        results['ids'][0],
        [metadata['source'] for metadata in results['metadatas'][0]],
        results['documents'][0],
        results['distances'][0],
    ):
        score = round(1 - float(distance), 4)
        if score > 0:
            hits.append({
                'id': int(chunk_id),
                'source': source,
                'text': text,
                'score': score,
            })
    return hits

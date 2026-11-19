import json
import math
import os
import re
import sqlite3
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parents[1]
DB = Path(os.environ.get('RAG_DB_PATH', str(BASE / 'knowledge.db')))


def chunks(text, size=95, overlap=20):
    words = text.split()
    if overlap >= size or size < 1:
        raise ValueError('size must exceed overlap')
    return [' '.join(words[i:i + size]) for i in range(0, len(words), size - overlap) if words[i:i + size]]


def connect(db=None):
    db = Path(db or DB)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute('CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, source TEXT NOT NULL, text TEXT NOT NULL)')
    return conn


def ingest(documents, db=None):
    with connect(db) as con:
        con.execute('DELETE FROM chunks')
        rows = [(d['source'], part) for d in documents for part in chunks(d['text'])]
        con.executemany('INSERT INTO chunks (source, text) VALUES (?, ?)', rows)
    return len(rows)


def ingest_demo(db=None):
    documents = load_demo()
    sources = {document['source'] for document in documents}
    rows = [
        (document['source'], part)
        for document in documents
        for part in chunks(document['text'])
    ]
    with connect(db) as con:
        placeholders = ','.join('?' for _ in sources)
        con.execute(
            f'DELETE FROM chunks WHERE source IN ({placeholders})',
            tuple(sources),
        )
        con.executemany('INSERT INTO chunks (source, text) VALUES (?, ?)', rows)
    return len(rows)


def retrieve(query, top_k=3, db=None):
    if not query.strip():
        return []
    with connect(db) as con:
        rows = con.execute('SELECT id, source, text FROM chunks ORDER BY id').fetchall()
    if not rows:
        return []
    documents = [row[2] for row in rows]
    words = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True,
    )
    characters = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        sublinear_tf=True,
    )
    word_matrix = words.fit_transform(documents)
    character_matrix = characters.fit_transform(documents)
    word_scores = cosine_similarity(words.transform([query]), word_matrix)[0]
    character_scores = cosine_similarity(
        characters.transform([query]),
        character_matrix,
    )[0]
    scores = 0.6 * word_scores + 0.4 * character_scores
    ranked = sorted(range(len(rows)), key=lambda i: float(scores[i]), reverse=True)
    return [
        {
            'id': rows[index][0],
            'source': rows[index][1],
            'text': rows[index][2],
            'score': round(float(scores[index]), 4),
        }
        for index in ranked[:top_k]
        if scores[index] > 0.02
    ]


def estimate_tokens(text):
    return math.ceil(len(text) / 4)


def inventory_reorder_tool(sku, available, threshold):
    available = max(0, int(available))
    threshold = int(threshold)
    if threshold < 0:
        raise ValueError('threshold must be nonnegative')
    return {'sku': sku, 'available': available, 'threshold': threshold, 'reorder_units': max(0, threshold - available)}


def answer(query, top_k=3, db=None):
    match = re.fullmatch(r'\s*reorder\s+([\w-]+)\s+available\s+(-?\d+)\s+threshold\s+(\d+)\s*', query, re.I)
    if match:
        return {'mode': 'tool', 'answer': inventory_reorder_tool(match.group(1), match.group(2), match.group(3)), 'citations': [], 'retrieved': 0, 'estimated_context_tokens': 0}
    hits = retrieve(query, top_k=top_k, db=db)
    if not hits:
        return {'mode': 'retrieval', 'answer': 'I could not find supporting evidence in the uploaded knowledge base.', 'citations': [], 'retrieved': 0, 'estimated_context_tokens': 0}
    citations = [{'source': h['source'], 'chunk_id': h['id'], 'score': h['score']} for h in hits]
    context = '\n'.join(h['text'] for h in hits)
    response = f'Best matching evidence from {hits[0]["source"]}: {hits[0]["text"]}'
    mode = 'extractive_rag'
    return {'mode': mode, 'answer': response, 'citations': citations, 'retrieved': len(hits), 'estimated_context_tokens': estimate_tokens(context)}


def load_demo():
    return json.loads((BASE / 'data' / 'demo_docs.json').read_text())

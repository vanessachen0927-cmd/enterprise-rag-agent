from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from app.config import MAX_DOCUMENT_CHARS, MAX_DOCUMENTS, PUBLIC_DEMO_MODE
from app.core import answer, connect, ingest, ingest_demo

app = FastAPI(title='Local Enterprise RAG & AI Agent', version='2.0')

class Question(BaseModel):
    question: str = Field(min_length=2, max_length=1200)
    top_k: int = Field(default=3, ge=1, le=8)

class Document(BaseModel):
    source: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=1000000)

@app.get('/health')
def health():
    with connect() as con:
        count = con.execute('SELECT count(*) FROM chunks').fetchone()[0]
        documents = con.execute('SELECT count(DISTINCT source) FROM chunks').fetchone()[0]
    from app.semantic import semantic_index_status

    return {
        'status': 'ok',
        'chunks': count,
        'documents': documents,
        'semantic_index': semantic_index_status(),
        'public_demo_mode': PUBLIC_DEMO_MODE,
    }

@app.post('/ingest')
def import_docs(documents: list[Document]):
    if not documents:
        raise HTTPException(400, 'Please provide at least one document')
    if len(documents) > MAX_DOCUMENTS:
        raise HTTPException(413, f'At most {MAX_DOCUMENTS} documents may be ingested')
    if any(len(document.text) > MAX_DOCUMENT_CHARS for document in documents):
        raise HTTPException(
            413,
            f'Document text may not exceed {MAX_DOCUMENT_CHARS} characters',
        )
    return {'chunks': ingest([d.model_dump() for d in documents])}

@app.post('/demo/load')
def demo():
    return {'chunks': ingest_demo()}

@app.post('/ask')
def ask(question: Question):
    if PUBLIC_DEMO_MODE and question.top_k > 3:
        raise HTTPException(422, 'top_k may not exceed 3 in public demo mode')
    return answer(question.question, top_k=question.top_k)

class AgentQuestion(Question):
    retrieval_mode: str = Field(default='lexical', pattern='^(lexical|semantic)$')

@app.post('/agent/run')
def agent_run(request: AgentQuestion):
    from app.agent import run_agent
    if PUBLIC_DEMO_MODE and request.top_k > 3:
        raise HTTPException(422, 'top_k may not exceed 3 in public demo mode')
    try:
        return run_agent(request.question, mode=request.retrieval_mode, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

@app.post('/semantic/index')
def semantic_index():
    if PUBLIC_DEMO_MODE:
        raise HTTPException(409, 'Local semantic indexing is disabled in public demo mode')
    from app.semantic import create_vector_index
    try:
        return create_vector_index()
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(503, f'Local semantic indexing failed: {exc}') from exc

@app.get('/evaluation')
def evaluation(
    mode: str = Query(default='lexical', pattern='^(lexical|semantic)$'),
    top_k: int = Query(default=1, ge=1, le=8),
):
    if PUBLIC_DEMO_MODE and (mode != 'lexical' or top_k > 3):
        raise HTTPException(409, 'Public demo evaluation is limited to lexical retrieval with top_k <= 3')
    from app.evaluate import evaluate_retrieval
    try:
        return evaluate_retrieval(top_k=top_k, mode=mode)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ImportError, OSError, RuntimeError) as exc:
        raise HTTPException(503, f'Retrieval evaluation failed: {exc}') from exc

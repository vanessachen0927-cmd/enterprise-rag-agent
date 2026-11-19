import json
import re
import time
from typing import Any, TypedDict

import httpx
import ollama
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from app.config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS, PUBLIC_DEMO_MODE
from app.core import inventory_reorder_tool, retrieve
from app.semantic import semantic_retrieve


class InventoryArguments(BaseModel):
    sku: str
    available: int
    threshold: int


class AgentState(TypedDict, total=False):
    messages: list[AnyMessage]
    question: str
    mode: str
    db: Any
    top_k: int
    embedder: Any
    llm_client: Any
    decision: dict[str, Any]
    result: dict[str, Any]


inventory_tool = StructuredTool.from_function(
    func=inventory_reorder_tool,
    name='inventory_reorder',
    description='Calculate read-only reorder units from available stock and a threshold.',
    args_schema=InventoryArguments,
)


def plan(question):
    match = re.fullmatch(
        r'\s*reorder\s+([\w-]+)\s+available\s+(-?\d+)\s+threshold\s+(\d+)\s*',
        question,
        re.I,
    )
    if match:
        return {
            'tool': 'inventory_reorder',
            'args': {
                'sku': match[1],
                'available': int(match[2]),
                'threshold': int(match[3]),
            },
        }
    return {'tool': 'knowledge_search', 'args': {'query': question}}


def _route(state):
    decision = plan(state['question'])
    result = {'decision': decision}
    if decision['tool'] == 'inventory_reorder':
        result['messages'] = [
            AIMessage(
                content='',
                tool_calls=[{
                    'name': inventory_tool.name,
                    'args': decision['args'],
                    'id': 'inventory_reorder_call',
                }],
            )
        ]
    return result


def _inventory_result(state):
    message = state['messages'][-1]
    result = json.loads(message.content) if isinstance(message.content, str) else message.content
    return {
        'result': {
            'mode': 'agent_tool',
            'answer': result,
            'citations': [],
            'retrieval_provider': 'LangGraph inventory tool',
            'trace': [
                {'step': 'route', 'tool': 'inventory_reorder'},
                {'step': 'execute', 'status': 'completed', 'writes': False},
            ],
            'requires_approval': result['reorder_units'] > 0,
        }
    }


def _knowledge_result(state):
    question = state['question']
    mode = state.get('mode', 'lexical')
    top_k = state.get('top_k', 3)
    fallback = None
    if mode == 'semantic':
        if PUBLIC_DEMO_MODE:
            hits = retrieve(question, top_k=top_k, db=state.get('db'))
            retrieval_mode = 'lexical_fallback'
            fallback = {
                'from': 'semantic',
                'to': 'lexical',
                'reason': 'disabled_in_public_demo',
            }
        else:
            try:
                hits = semantic_retrieve(
                    question,
                    top_k=top_k,
                    db=state.get('db'),
                    embedder=state.get('embedder'),
                )
                retrieval_mode = 'semantic'
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                hits = retrieve(question, top_k=top_k, db=state.get('db'))
                retrieval_mode = 'lexical_fallback'
                fallback = {'from': 'semantic', 'to': 'lexical', 'reason': str(exc)}
    elif mode == 'lexical':
        hits = retrieve(question, top_k=top_k, db=state.get('db'))
        retrieval_mode = 'lexical'
    else:
        raise ValueError('mode must be lexical or semantic')
    trace = [
        {'step': 'route', 'tool': 'knowledge_search'},
        {'step': 'retrieve', 'mode': retrieval_mode, 'citations': len(hits)},
    ]
    if not hits:
        response = {
            'mode': retrieval_mode,
            'answer': 'I could not find supporting evidence in the local knowledge base.',
            'citations': [],
            'retrieval_provider': (
                'ChromaDB / sentence-transformers'
                if retrieval_mode == 'semantic'
                else 'TF-IDF lexical'
            ),
            'trace': trace + [{'step': 'ground', 'status': 'abstained'}],
            'requires_approval': False,
        }
        if fallback:
            response['fallback'] = fallback
        return {'result': response}
    citations = [
        {'source': hit['source'], 'chunk_id': hit['id'], 'score': hit['score']}
        for hit in hits
    ]
    evidence = '\n'.join(f'[{hit["source"]}] {hit["text"]}' for hit in hits)
    answer_text = f'Best matching evidence from {hits[0]["source"]}: {hits[0]["text"]}'
    llm_status = 'disabled_in_public_demo' if PUBLIC_DEMO_MODE else 'extractive_fallback'
    generation_error_type = None
    if not PUBLIC_DEMO_MODE:
        try:
            client = state.get('llm_client')
            if client is None:
                client = ollama.Client(
                    host=OLLAMA_HOST,
                    timeout=OLLAMA_TIMEOUT_SECONDS,
                )
            reply = client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {
                        'role': 'system',
                        'content': (
                            'Answer only from the provided policy excerpts. '
                            'Include the source filename. If evidence is insufficient, '
                            'say so. Do not invent policy details.'
                        ),
                    },
                    {'role': 'user', 'content': f'Question: {question}\n\nEvidence:\n{evidence}'},
                ],
                options={'temperature': 0},
            )
            generated = reply.message.content.strip()
            if generated:
                answer_text = generated
                llm_status = 'ollama'
            else:
                llm_status = 'ollama_empty_response'
        except (ConnectionError, httpx.HTTPError, ollama.ResponseError) as exc:
            llm_status = 'ollama_unavailable'
            generation_error_type = type(exc).__name__
    generation_step = {'step': 'generate', 'status': llm_status}
    if generation_error_type:
        generation_step['error_type'] = generation_error_type
    trace.append(generation_step)
    trace.append({'step': 'ground', 'status': 'supported'})
    response = {
        'mode': f'{retrieval_mode}_rag',
        'answer': answer_text,
        'citations': citations,
        'retrieval_provider': (
            'ChromaDB / sentence-transformers'
            if retrieval_mode == 'semantic'
            else 'TF-IDF lexical'
        ),
        'trace': trace,
        'requires_approval': False,
    }
    if fallback:
        response['fallback'] = fallback
    return {'result': response}


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node('route', _route)
    graph.add_node('tools', ToolNode([inventory_tool]))
    graph.add_node('inventory_result', _inventory_result)
    graph.add_node('knowledge_search', _knowledge_result)
    graph.add_edge(START, 'route')
    graph.add_conditional_edges(
        'route',
        lambda state: 'tools' if state['decision']['tool'] == 'inventory_reorder' else 'knowledge_search',
        {'tools': 'tools', 'knowledge_search': 'knowledge_search'},
    )
    graph.add_edge('tools', 'inventory_result')
    graph.add_edge('inventory_result', END)
    graph.add_edge('knowledge_search', END)
    return graph.compile()


agent_graph = _build_graph()


def run_agent(question, mode='lexical', db=None, top_k=3, embedder=None, llm_client=None):
    started = time.perf_counter()
    state = agent_graph.invoke({
        'question': question,
        'mode': mode,
        'db': db,
        'top_k': top_k,
        'embedder': embedder,
        'llm_client': llm_client,
    })
    result = state['result']
    result['latency_ms'] = round((time.perf_counter() - started) * 1000, 3)
    return result

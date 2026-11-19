import requests
import streamlit as st

st.set_page_config(page_title='Local Enterprise RAG Agent')
st.title('Enterprise Knowledge Agent')
st.caption('Local retrieval, cited policy answers, and a read-only inventory tool')

api_url = st.sidebar.text_input('API URL', 'http://127.0.0.1:8000').rstrip('/')


def api_request(method, path, payload=None, params=None, timeout=30):
    try:
        response = requests.request(
            method,
            f'{api_url}{path}',
            json=payload,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        if path == '/evaluation' and exc.response.status_code == 409:
            try:
                detail = exc.response.json().get('detail', exc.response.text)
            except ValueError:
                detail = exc.response.text
            st.warning(f'Evaluation unavailable: {detail}')
        else:
            st.error(f'API returned {exc.response.status_code}: {exc.response.text}')
    except requests.RequestException as exc:
        st.error(f'Could not reach the API: {exc}')
    return None


health = api_request('GET', '/health', timeout=5)
public_mode = bool(health and health.get('public_demo_mode'))
if health:
    st.sidebar.success('API ready')
    if public_mode:
        st.sidebar.info('Public demo mode: bounded ingestion, lexical retrieval, no local LLM.')
else:
    st.sidebar.warning('API is unavailable')

if st.sidebar.button('Load sample policies'):
    loaded = api_request('POST', '/demo/load')
    if loaded:
        st.sidebar.success(f'Loaded {loaded["chunks"]} chunks')

if not public_mode and st.sidebar.button('Build/update local semantic index'):
    with st.spinner('Loading the local embedding model and indexing documents…'):
        indexed = api_request('POST', '/semantic/index', timeout=300)
    if indexed:
        st.sidebar.success(f'Indexed {indexed["vectors"]} chunks with {indexed["model"]}')

health = api_request('GET', '/health', timeout=5) or health
if health:
    st.sidebar.metric('Loaded documents', health['documents'])
    index_status = health['semantic_index']
    st.sidebar.write(f'Semantic index: **{index_status["status"]}**')
    if index_status['status'] != 'not_built':
        st.sidebar.caption(
            f'{index_status["indexed_chunks"]} chunks · {index_status["model"]}'
        )
provider_indicator = st.sidebar.empty()
provider_indicator.markdown(
    'Last retrieval provider: **'
    + st.session_state.get('last_retrieval_provider', 'Not used yet')
    + '**'
)

with st.expander('Ingest a policy document'):
    with st.form('document_ingest'):
        source = st.text_input('Source name', 'new_policy.md')
        text = st.text_area('Document text', height=160)
        submitted = st.form_submit_button('Ingest document')
    if submitted:
        if not source.strip() or not text.strip():
            st.error('Provide both a source name and document text.')
        else:
            ingested = api_request(
                'POST',
                '/ingest',
                [{'source': source.strip(), 'text': text}],
                timeout=30,
            )
            if ingested:
                st.success(f'Ingested {ingested["chunks"]} chunks. Rebuild the semantic index if using semantic retrieval.')

st.subheader('Ask the agent')
with st.form('agent_query'):
    question = st.text_input(
        'Question or inventory command',
        'When can a customer return an item?',
    )
    mode_options = ['lexical'] if public_mode else ['lexical', 'semantic']
    mode = st.selectbox('Retrieval mode', mode_options)
    top_k = st.slider('Maximum sources', min_value=1, max_value=3 if public_mode else 8, value=3)
    ask = st.form_submit_button('Run agent')
if ask and question.strip():
    result = api_request(
        'POST',
        '/agent/run',
        {'question': question, 'retrieval_mode': mode, 'top_k': top_k},
        timeout=120,
    )
    if result:
        st.session_state['last_retrieval_provider'] = result.get(
            'retrieval_provider',
            result.get('mode', 'unknown'),
        )
        provider_indicator.markdown(
            'Last retrieval provider: **'
            + st.session_state['last_retrieval_provider']
            + '**'
        )
        st.write(result['answer'])
        st.write('Source citations')
        if result.get('citations'):
            for citation in result['citations']:
                st.write(
                    f'{citation["source"]} - chunk {citation["chunk_id"]} - '
                    f'score {citation["score"]}'
                )
        else:
            st.write('No document citations for this result.')
        st.caption(f'Latency: {result.get("latency_ms", 0)} ms')
        if result.get('fallback'):
            st.info(f'Fallback: {result["fallback"]["reason"]}')
        with st.expander('Agent trace'):
            st.json(result.get('trace', []))
        if result.get('requires_approval'):
            st.warning('This proposed reorder requires human approval. No inventory was changed.')

st.subheader('Retrieval evaluation')
evaluation_mode = st.selectbox(
    'Evaluation retrieval mode',
    ['lexical'] if public_mode else ['lexical', 'semantic'],
)
if st.button('Run evaluation'):
    report = api_request(
        'GET',
        '/evaluation',
        params={'mode': evaluation_mode, 'top_k': 1},
        timeout=120,
    )
    if report:
        st.metric('Recall@1', report['recall_at_k'])
        st.write(
            f'{report["hits"]}/{report["total"]} relevant sources - '
            f'mean {report["mean_latency_ms"]} ms - '
            f'p95 {report["p95_latency_ms"]} ms'
        )
        st.dataframe(report['cases'], use_container_width=True)

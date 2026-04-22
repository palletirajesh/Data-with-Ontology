import os
import time
import streamlit as st
import pandas as pd
import requests
import rdflib
from google.cloud import bigquery
from google.oauth2 import service_account
from sentence_transformers import SentenceTransformer, util

# --- 1. CONFIG & STYLE ---
st.set_page_config(page_title="Risk Data Agent", page_icon="🏦", layout="wide")

# Professional Fonts and CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .stDownloadButton>button { width: 100%; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
BQ_PROJECT = st.secrets["bigquery"]["project_id"]
BQ_DATASET = st.secrets["bigquery"]["dataset_id"]

# --- 2. ENGINES ---
@st.cache_resource
def get_bq_client():
    info = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(info)
    return bigquery.Client(credentials=credentials, project=BQ_PROJECT)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

bq_client = get_bq_client()
embedder = load_embedding_model()

# --- 3. KNOWLEDGE & SEMANTIC LOGIC ---
@st.cache_data
def get_unified_knowledge():
    knowledge_chunks = []
    try:
        with open("database_schema (3).md", "r") as f:
            content = f.read()
            knowledge_chunks.extend(["TABLE_STRUCT: " + t for t in content.split("### TABLE:")[1:]])
    except: pass

    try:
        g = rdflib.Graph().parse("knowledge_base (3).jsonld", format="json-ld")
        q_joins = "SELECT ?tLabel ?sK ?tK WHERE { ?j a <http://com/ontology#JoinDefinition> ; <http://com/ontology#targetTable> ?t ; <http://com/ontology#sourceKey> ?sK ; <http://com/ontology#targetKey> ?tK . ?t <http://www.w3.org/2000/01/rdf-schema#label> ?tLabel . }"
        for row in g.query(q_joins):
            knowledge_chunks.append(f"MANDATORY JOIN: Join to {row.tLabel} ON {row.sK} = {row.tK}")
        
        q_jargon = "SELECT ?colLabel ?jargon WHERE { ?col <http://www.w3.org/2000/01/rdf-schema#label> ?colLabel ; <http://com/ontology#businessJargon> ?jargon . }"
        for row in g.query(q_jargon):
            knowledge_chunks.append(f"BUSINESS TRANSLATION: Term '{row.jargon}' maps to column '{row.colLabel}'")
    except: pass
    return knowledge_chunks

def get_semantic_context(query, knowledge_base, top_k=8):
    query_emb = embedder.encode(f"search_query: {query}", convert_to_tensor=True)
    kb_embs = embedder.encode(knowledge_base, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, kb_embs, top_k=top_k)
    return "\n\n".join([knowledge_base[hit['corpus_id']] for hit in hits[0]])

# --- 4. DATA DOWNLOADER ---
@st.cache_data
def get_full_dataset_csv():
    # Pulls a small sample so users can understand the DB structure
    query = f"SELECT * FROM `{BQ_PROJECT}.{BQ_DATASET}.dim_customer` LIMIT 100"
    return bq_client.query(query).to_dataframe().to_csv(index=False)

# --- 5. INTELLIGENT SQL GENERATION ---
def generate_sql(user_query, context):
    full_path = f"{BQ_PROJECT}.{BQ_DATASET}"
    
    system_prompt = f"""You are a BigQuery SQL Expert. 
    Context: {context}
    
    STRICT RULES:
    1. ONLY use Tables/Columns in Context.
    2. Table Format: ALWAYS use backticks and the full path: `{full_path}.table_name`
    3. NO PARENTHESES: Never put () after a table name.
    4. Use MANDATORY JOINs exactly. Sequence tables logically.
    5. Output ONLY raw SQL code. No markdown or explanations.
    6. If data is missing, output EXACTLY: "I cannot answer this with the available data."
    7. Always begin with a SELECT clause. For multi-table joins, dim_customer should act a bridge table
    8. EVERY table name in the SQL must be prefixed with: `{full_path}.
    9. Example format: SELECT * FROM `{full_path}.dim_customer` JOIN `{full_path}.dim_card_association`
    10. Tables available: dim_customer, dim_card_association, fact_card_ledger, fact_credit_bureau.
    11. For string filters, use: UPPER(column) = UPPER('value').
    12. TRANSLATION: Apply BUSINESS TRANSLATION RULES strictly to map user jargon to correct columns.
    13. GRANULARITY: Unless the user explicitly uses words like 'count', 'total', or 'how many', ALWAYS return a detailed list of records (SELECT *) rather than a summary or count.
    """
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": "llama-3.3-70b-versatile", "messages": messages, "temperature": 0}
    ).json()
    
    return response['choices'][0]['message']['content'].strip().replace("```sql", "").replace("```", "").strip()

# --- 6. SIDEBAR HISTORY ---
with st.sidebar:
    st.title("🕒 Query History")
    try:
        # Load from your query_history table
        hist_query = f"SELECT user_query FROM `{BQ_PROJECT}.{BQ_DATASET}.query_history` ORDER BY created_at DESC LIMIT 8"
        hist_df = bq_client.query(hist_query).to_dataframe()
        for q in hist_df['user_query']:
            st.caption(f"🗨️ {q}")
            st.divider()
    except:
        st.write("No history available yet.")

# --- 7. MAIN UI ---
st.title("🏦 Risk Data Intelligence Agent")

# Requirement 2: Small download button above input
col_dl, _ = st.columns([1, 3])
with col_dl:
    st.download_button(
        label="📥 Download Data Sample",
        data=get_full_dataset_csv(),
        file_name="risk_data_dictionary.csv",
        mime="text/csv",
    )

user_input = st.text_input("Analyze your risk data:", placeholder="e.g. Names of clients with FICO scores > 750")

if user_input:
    # Requirement 6: Clear old results by not displaying session_state until button click
    kb = get_unified_knowledge()
    context = get_semantic_context(user_input, kb)
    generated_sql = generate_sql(user_input, context)

    col_sql, col_res = st.columns([4, 6])
    
    with col_sql:
        st.subheader("📝 Generated SQL")
        final_sql = st.text_area("Review AI Logic:", value=generated_sql, height=250)
        
        # Requirement 5: Execution state and timer
        if st.button("▶️ Execute Query"):
            start_time = time.perf_counter()
            with st.status("🚀 Running BigQuery Job...", expanded=True) as status:
                try:
                    df = bq_client.query(final_sql).to_dataframe()
                    end_time = time.perf_counter()
                    st.session_state.last_df = df
                    status.update(label=f"✅ Query Finished in {end_time - start_time:.2f}s", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="❌ Query Error", state="error")
                    st.error(e)

    with col_res:
        st.subheader("📊 Data Results")
        if "last_df" in st.session_state:
            st.dataframe(st.session_state.last_df, use_container_width=True)

# Requirement 4: Expanded Architecture Details
st.divider()
st.subheader("🏗️ System Architecture")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.info("**1. User Interface**\nStreamlit collects natural language inputs and provides contextual data downloads.")
with c2:
    st.info("**2. Semantic Layer**\nInput is passed through **MiniLM Embeddings** to map jargon against the **JSON-LD Ontology**.")
with c3:
    st.info("**3. Logic Synthesis**\n**Llama-3.3-70B** reads the combined schema/ontology to engineer BigQuery-compliant SQL.")
with c4:
    st.info("**4. Cloud Warehouse**\n**Google BigQuery** executes serverless queries, returning production-grade results to the UI.")

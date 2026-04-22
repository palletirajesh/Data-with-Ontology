import os
# CRITICAL FIX FOR STREAMLIT CLOUD
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import requests
import rdflib
from google.cloud import bigquery
from google.oauth2 import service_account
from sentence_transformers import SentenceTransformer, util

# ==========================================
# --- 1. CONFIGURATION & SECRETS ---
# ==========================================
st.set_page_config(page_title="Risk Data Agent", page_icon="🏦", layout="wide")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
BQ_PROJECT = st.secrets["bigquery"]["project_id"]
BQ_DATASET = st.secrets["bigquery"]["dataset_id"]

@st.cache_resource
def get_bq_client():
    """Initializes BigQuery client using Streamlit secrets."""
    credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
    return bigquery.Client(credentials=credentials, project=BQ_PROJECT)

bq_client = get_bq_client()

# ==========================================
# --- 2. CORE ENGINES (EMBEDDINGS) ---
# ==========================================
@st.cache_resource
def load_embedding_model():
    """Lightweight 80MB embeddings (MiniLM)."""
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedding_model()

# ==========================================
# --- 3. GLOBAL HISTORY ENGINE ---
# ==========================================
def save_to_global_history(query, sql_code):
    """Saves telemetry and query history to BigQuery."""
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.query_history"
    rows_to_insert = [
        {"user_query": query, "generated_sql": sql_code}
    ]
    try:
        bq_client.insert_rows_json(table_id, rows_to_insert)
    except: pass # Silent fail if history table isn't created yet

def load_global_history(limit=5):
    """Loads history from the BigQuery table."""
    query = f"SELECT user_query as query, generated_sql as sql FROM `{BQ_PROJECT}.{BQ_DATASET}.query_history` ORDER BY created_at DESC LIMIT {limit}"
    try:
        return bq_client.query(query).to_dataframe().to_dict('records')
    except: return []

# ==========================================
# --- 4. UNIFIED CONTEXT ENGINE ---
# ==========================================
@st.cache_data
def get_unified_knowledge():
    """Combines Schema (MD) and Ontology (JSON-LD) into searchable chunks."""
    knowledge_chunks = []
    
    # 1. Technical Schema
    try:
        with open("database_schema.md", "r") as f:
            content = f.read()
            knowledge_chunks.extend(["TABLE_STRUCT: " + t for t in content.split("### TABLE:")[1:]])
    except: pass

    # 2. Joins & Jargon from JSON-LD
    try:
        g = rdflib.Graph().parse("knowledge_base.jsonld", format="json-ld")
        
        # Extract Joins
        q_joins = "SELECT ?tLabel ?sK ?tK WHERE { ?j a <http://com/ontology#JoinDefinition> ; <http://com/ontology#targetTable> ?t ; <http://com/ontology#sourceKey> ?sK ; <http://com/ontology#targetKey> ?tK . ?t <http://www.w3.org/2000/01/rdf-schema#label> ?tLabel . }"
        for row in g.query(q_joins):
            knowledge_chunks.append(f"MANDATORY JOIN: Join to {row.tLabel} ON {row.sK} = {row.tK}")
            
        # Extract Jargon
        q_jargon = "SELECT ?colLabel ?jargon WHERE { ?col <http://www.w3.org/2000/01/rdf-schema#label> ?colLabel ; <http://com/ontology#businessJargon> ?jargon . }"
        for row in g.query(q_jargon):
            knowledge_chunks.append(f"BUSINESS TRANSLATION: User term '{row.jargon}' maps to column '{row.colLabel}'")
    except: pass
    
    return knowledge_chunks

def get_semantic_context(query, knowledge_base, top_k=8):
    """RAG-style retrieval to find relevant rules for the query."""
    query_emb = embedder.encode(f"search_query: {query}", convert_to_tensor=True)
    kb_embs = embedder.encode(knowledge_base, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, kb_embs, top_k=top_k)
    return "\n\n".join([knowledge_base[hit['corpus_id']] for hit in hits[0]])

# ==========================================
# --- 5. THE SQL ENGINEER (PROMPT) ---
# ==========================================
def generate_sql(user_query, context, history):
    project_id = st.secrets["bigquery"]["project_id"]
    dataset_id = st.secrets["bigquery"]["dataset_id"]
    
    # Create the prefix string
    full_prefix = f"{project_id}.{dataset_id}"

    system_prompt = f"""You are a Google BigQuery SQL Expert. Context:
    {context}
        
    STRICT RULES:
    1. ONLY use Tables/Columns in Context.
    2. Use MANDATORY JOINs exactly. Sequence tables logically.
    3. Output ONLY raw SQL code. No markdown or explanations.
    4. If data is missing, output EXACTLY: "I cannot answer this with the available data."
    5. Always begin with a SELECT clause. For multi-table joins, the main FROM table MUST be 'gen_ai_bank.dim_customer'.
    6. For string filters, use: UPPER(column) = UPPER('value').
    7. TRANSLATION: Apply BUSINESS TRANSLATION RULES strictly to map user jargon to correct columns.
    8. GRANULARITY: Unless the user explicitly uses words like 'count', 'total', or 'how many', ALWAYS return a detailed list of records (SELECT *) rather than a summary or count.
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-2:]:
        messages.append({"role": "user", "content": h['query']})
        messages.append({"role": "assistant", "content": h['sql']})
    messages.append({"role": "user", "content": user_query})

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": "llama-3.3-70b-versatile", "messages": messages, "temperature": 0.1}
    ).json()
    
    if "choices" in response:
        return response['choices'][0]['message']['content'].strip().replace("```sql", "").replace("```", "").strip()
    return "I cannot answer this with available data."
    forbidden_words = ["DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE", "ALTER"]
    
    if any(word in generated_sql.upper() for word in forbidden_words):
        st.error("⚠️ Security Alert: Destructive SQL command detected. Query blocked.")
        return "SELECT 'Access Denied' as status"
    
    return generated_sql
# ==========================================
# --- 6. UI & EXECUTION ---
# ==========================================
st.title("🏦 Risk Data Agent (BigQuery)")
user_input = st.text_input("Ask a question about customers or credit data:")

kb = get_unified_knowledge()
history = load_global_history()

if user_input:
    context = get_semantic_context(user_input, kb)
    generated_sql = generate_sql(user_input, context, history)
    
    col1, col2 = st.columns([4, 6])
    with col1:
        st.markdown("### 📝 BigQuery SQL")
        final_sql = st.text_area("Draft:", value=generated_sql, height=250)
        if st.button("▶️ Execute Query", type="primary"):
            try:
                # BigQuery execution
                df = bq_client.query(final_sql).to_dataframe()
                st.session_state.df = df
                save_to_global_history(user_input, final_sql)
                st.rerun()
            except Exception as e: st.error(f"BQ Error: {e}")

    with col2:
        st.markdown("### 📊 Results")
        if "df" in st.session_state:
            st.dataframe(st.session_state.df, use_container_width=True)

# Technical Rationale Section
st.divider()
st.subheader("🛠️ Technical Architecture")
t1, t2, t3, t4 = st.columns(4)
with t1:
    st.write("**🧠 Llama-3.3-70B**\nGroq-powered SQL Generation.")
with t2:
    st.write("**🔍 Semantic Retrieval**\nMiniLM embeddings for accurate mapping.")
with t3:
    st.write("**📑 BigQuery Warehouse**\nProduction-scale serverless analytics.")
with t4:
    st.write("**🏗️ Global History**\nShared persistence across user sessions.")

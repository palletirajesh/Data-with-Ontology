import os
# CRITICAL FIX FOR STREAMLIT CLOUD SEGFAULT
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import requests
import rdflib
import duckdb  # Switched from databricks.sql
from sentence_transformers import SentenceTransformer, util

# ==========================================
# --- 1. CONFIGURATION ---
# ==========================================
st.set_page_config(page_title="Risk Data Agent", page_icon="🏦", layout="wide")

# Ensure you have GROQ_API_KEY in your Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ==========================================
# --- 2. DUCKDB ENGINE (FREE & FAST) ---
# ==========================================
@st.cache_resource
def load_embedding_model():
    """Lightweight 80MB embeddings (MiniLM)."""
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedding_model()

@st.cache_resource
def get_db_connection():
    """Initializes DuckDB and maps your Parquet files to SQL tables."""
    try:
        # Connect to a local file database so history persists
        conn = duckdb.connect(database='bank_data.db', read_only=False)
        
        # Map your 4 Parquet files as SQL Views
        tables = [
            'dim_card_association', 
            'fact_credit_bureau', 
            'fact_card_ledger', 
            'dim_customer'
        ]
        
        for table in tables:
            # This makes the .parquet file queryable as a table name
            conn.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM '{table}.parquet'")
        
        # Initialize internal history table if it doesn't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_query TEXT,
                generated_sql TEXT,
                user_location TEXT
            )
        """)
        return conn
    except Exception as e:
        st.error(f"❌ Database Engine Error: {e}")
        return None

conn = get_db_connection()

# ==========================================
# --- 3. TELEMETRY & HISTORY ---
# ==========================================
@st.cache_data(ttl=3600)
def get_client_location():
    try:
        headers = st.context.headers
        ip = headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if not ip or ip in ["127.0.0.1", "localhost"]: return "Local Network"
        response = requests.get(f"http://ip-api.com/json/{ip}", timeout=2)
        data = response.json()
        return f"{data['city']}, {data['country']}" if data.get("status") == "success" else "Unknown"
    except: return "Unknown"

def save_to_global_history(query, sql_code):
    location = get_client_location()
    conn.execute(
        "INSERT INTO query_history (user_query, generated_sql, user_location) VALUES (?, ?, ?)", 
        [query, sql_code, location]
    )

def load_global_history(limit=10):
    try:
        return conn.execute(f"""
            SELECT user_location, user_query as query, generated_sql as sql 
            FROM query_history 
            ORDER BY created_at DESC LIMIT {limit}
        """).df().to_dict('records')
    except: return []

# ==========================================
# --- 4. UNIFIED CONTEXT ENGINE ---
# ==========================================
@st.cache_data
def get_unified_knowledge():
    knowledge_chunks = []
    # Load schema from MD
    try:
        with open("database_schema.md", "r") as f:
            content = f.read()
            knowledge_chunks.extend(["TABLE_STRUCT: " + t for t in content.split("### TABLE:")[1:]])
    except: pass
    
    # Parse JSON-LD Ontology
    try:
        g = rdflib.Graph().parse("knowledge_base.jsonld", format="json-ld")
        # Extract Join Rules
        q_joins = "SELECT ?tLabel ?sK ?tK WHERE { ?j a <http://gen_ai_bank.com/ontology#JoinDefinition> ; <http://gen_ai_bank.com/ontology#targetTable> ?t ; <http://gen_ai_bank.com/ontology#sourceKey> ?sK ; <http://gen_ai_bank.com/ontology#targetKey> ?tK . ?t rdfs:label ?tLabel . }"
        for row in g.query(q_joins):
            knowledge_chunks.append(f"MANDATORY JOIN: Join to {row.tLabel} ON {row.sK} = {row.tK}")
        
        # Extract Jargon
        q_jargon = "SELECT ?colLabel ?jargon WHERE { ?col a <http://gen_ai_bank.com/ontology#Column> ; <http://www.w3.org/2000/01/rdf-schema#label> ?colLabel ; <http://gen_ai_bank.com/ontology#representsConcept> ?concept . ?concept <http://gen_ai_bank.com/ontology#businessJargon> ?jargon . }"
        for row in g.query(q_jargon):
            knowledge_chunks.append(f"BUSINESS TRANSLATION: User term '{row.jargon}' maps to column '{row.colLabel}'")
    except: pass
    return knowledge_chunks

def get_semantic_context(query, knowledge_base, top_k=10):
    query_emb = embedder.encode(f"search_query: {query}", convert_to_tensor=True)
    kb_embs = embedder.encode(knowledge_base, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, kb_embs, top_k=top_k)
    return "\n\n".join([knowledge_base[hit['corpus_id']] for hit in hits[0]])

# ==========================================
# --- 5. THE SQL ENGINEER (GROQ) ---
# ==========================================
def generate_sql(user_query, context, history):
    system_prompt = f"""You are a DuckDB SQL Expert. Context:
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

    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": "llama-3.3-70b-versatile", "messages": messages, "temperature": 0.1}
    ).json()
    
    if "choices" in res:
        return res['choices'][0]['message']['content'].strip().replace("```sql", "").replace("```", "").strip()
    return "I cannot answer this with available data."

# ==========================================
# --- 6. UI & EXECUTION ---
# ==========================================
st.title("🏦 Risk Data Agent (Edge-AI)")
st.info("System running on DuckDB. Zero cloud costs active.")

user_input = st.text_input("Query Bank Data:", placeholder="e.g. Which clients have an OSUC > 5000?")

kb = get_unified_knowledge()
history = load_global_history()

if user_input:
    context = get_semantic_context(user_input, kb)
    generated_sql = generate_sql(user_input, context, history)
    
    col_l, col_r = st.columns([4, 6])
    with col_l:
        st.markdown("### 📝 SQL Draft")
        final_sql = st.text_area("SQL:", value=generated_sql, height=250)
        if st.button("▶️ Run Query", type="primary", use_container_width=True):
            try:
                # DuckDB returns a dataframe directly via .df()
                res_df = conn.execute(final_sql).df()
                st.session_state.df = res_df
                save_to_global_history(user_input, final_sql)
                st.rerun()
            except Exception as e: st.error(f"SQL Error: {e}")

    with col_r:
        st.markdown("### 📊 Results")
        if "df" in st.session_state:
            st.dataframe(st.session_state.df, use_container_width=True)

# ==========================================
# --- Sidebar History ---
# ==========================================
with st.sidebar:
    st.header("🌐 Global Query Memory")
    for h in history:
        st.info(f"📍 {h['user_location']}\n🗨️ {h['query']}")

# ==========================================
# --- 7. TECHNICAL RATIONALE ---
# ==========================================
st.divider()
st.subheader("🛠️ Technical Architecture")
t1, t2, t3, t4 = st.columns(4)
with t1:
    st.write("**🧠 Llama-3.3-70B**\nSQL Engineering at sub-second speeds.")
with t2:
    st.write("**🔍 DuckDB Engine**\nZero-cost, in-process analytics via Parquet.")
with t3:
    st.write("**📑 Ontology Layer**\nJSON-LD enforces join logic and business jargon.")
with t4:
    st.write("**🚀 Edge Architecture**\nNo external SQL warehouse latency.")

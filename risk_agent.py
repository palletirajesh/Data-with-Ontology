import os
# CRITICAL FIX FOR STREAMLIT CLOUD SEGFAULT
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import requests
import rdflib
from databricks import sql
from sentence_transformers import SentenceTransformer, util

# ==========================================
# --- 1. CONFIGURATION & SECRETS ---
# ==========================================
st.set_page_config(page_title="Risk Data Agent", page_icon="🏦", layout="wide")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
DB_CONFIG = {
    "server_hostname": st.secrets["DB_SERVER_HOSTNAME"], 
    "http_path": st.secrets["DB_HTTP_PATH"], 
    "access_token": st.secrets["DB_ACCESS_TOKEN"]
}

# ==========================================
# --- 2. CORE ENGINES (EMBEDDINGS & DB) ---
# ==========================================
@st.cache_resource
def load_embedding_model():
    """Lightweight 80MB embeddings (MiniLM) that fit in Streamlit's 1GB limit."""
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedding_model()

# We deliberately DO NOT use caching here so Databricks sessions don't go stale
# ==========================================
# --- 2. MULTI-TIER CONNECTION ENGINE ---
# ==========================================

def get_db_connection():
    """Connects to a local DuckDB instance (In-Process)."""
    try:
        # This creates an in-memory database
        conn = duckdb.connect(database=':memory:')
        
        # 'Register' your parquet files as if they were tables
        conn.execute("CREATE VIEW dim_customer AS SELECT * FROM 'dim_customer.parquet'")
        conn.execute("CREATE VIEW fact_risk AS SELECT * FROM 'fact_risk.parquet'")
        # Add any other tables here...
        
        return conn
    except Exception as e:
        st.error(f"❌ DuckDB Loading Error: {e}")
        return None

# The rest of your code remains the same! 
# DuckDB uses the same SQL syntax as Databricks for 99% of queries.

# --- INVISIBLE TELEMETRY ENGINE ---
@st.cache_data(ttl=3600)
def get_client_location():
    """Silently captures City and Country based on browser headers."""
    try:
        headers = st.context.headers
        ip = headers.get("X-Forwarded-For", "").split(",")[0].strip()
        
        if not ip or ip == "127.0.0.1" or ip == "localhost":
            return "Local Network"
            
        response = requests.get(f"http://ip-api.com/json/{ip}", timeout=2)
        data = response.json()
        
        if data.get("status") == "success":
            return f"{data['city']}, {data['country']}"
        return "Unknown Location"
    except Exception:
        return "Unknown Location"

# ==========================================
# --- 3. GLOBAL SLIDING WINDOW (MEMORY) ---
# ==========================================
def init_global_history():
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM gen_ai_bank.query_history LIMIT 1")
    except Exception:
        st.sidebar.error("⚠️ Global History Table not accessible in Databricks.")

def save_to_global_history(query, sql_code):
    location = get_client_location()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO gen_ai_bank.query_history (user_query, generated_sql, user_location) VALUES (?, ?, ?)", 
                [query, sql_code, location]
            )
    except:
        pass

def load_global_history(limit=10):
    try:
        return pd.read_sql(f"SELECT user_location, user_query as query, generated_sql as sql FROM gen_ai_bank.query_history ORDER BY created_at DESC LIMIT {limit}", conn).to_dict('records')
    except: return []

# ==========================================
# --- 4. UNIFIED CONTEXT ENGINE ---
# ==========================================
@st.cache_data
def get_unified_knowledge():
    knowledge_chunks = []
    
    # 1. Load technical schema from MD
    try:
        with open("database_schema.md", "r") as f:
            content = f.read()
            knowledge_chunks.extend(["TABLE_STRUCT: " + t for t in content.split("### TABLE:")[1:]])
    except FileNotFoundError:
        pass
    
    # Parse the JSON-LD Knowledge Base
    g = rdflib.Graph().parse("knowledge_base.jsonld", format="json-ld")
    
    # 2. Load mandatory Join Rules from JSONLD
    q_joins = """
    SELECT ?tLabel ?sK ?tK WHERE { 
        ?j a <http://gen_ai_bank.com/ontology#JoinDefinition> ; 
           <http://gen_ai_bank.com/ontology#targetTable> ?t ; 
           <http://gen_ai_bank.com/ontology#sourceKey> ?sK ; 
           <http://gen_ai_bank.com/ontology#targetKey> ?tK . 
        ?t rdfs:label ?tLabel . 
    }"""
    for row in g.query(q_joins):
        knowledge_chunks.append(f"MANDATORY JOIN: Join to {row.tLabel} ON {row.sK} = {row.tK}")
        
    # 3. Load Business Jargon & Concepts from JSONLD
    q_jargon = """
    SELECT ?colLabel ?jargon WHERE { 
        ?col a <http://gen_ai_bank.com/ontology#Column> ; 
             <http://www.w3.org/2000/01/rdf-schema#label> ?colLabel ; 
             <http://gen_ai_bank.com/ontology#representsConcept> ?concept . 
        ?concept <http://gen_ai_bank.com/ontology#businessJargon> ?jargon . 
    }"""
    
    jargon_mapping = {}
    for row in g.query(q_jargon):
        col_name = str(row.colLabel)
        jargon_term = str(row.jargon)
        
        if col_name not in jargon_mapping:
            jargon_mapping[col_name] = []
        jargon_mapping[col_name].append(jargon_term)
        
    for col, jargons in jargon_mapping.items():
        jargon_list_str = ", ".join(f"'{j}'" for j in jargons)
        knowledge_chunks.append(
            f"BUSINESS TRANSLATION RULE: If the user asks about {jargon_list_str}, they are referring to the database column '{col}'."
        )
            
    return knowledge_chunks

def get_semantic_context(query, knowledge_base, top_k=10):
    query_emb = embedder.encode(f"search_query: {query}", convert_to_tensor=True)
    kb_embs = embedder.encode(knowledge_base, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, kb_embs, top_k=top_k)
    return "\n\n".join([knowledge_base[hit['corpus_id']] for hit in hits[0]])

# ==========================================
# --- 5. THE SQL ENGINEER (GROQ LLAMA-3.3) ---
# ==========================================
def generate_sql(user_query, context, history):
    system_prompt = f"""You are a Databricks SQL Expert. Context:
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
    for h in history[-3:]:
        messages.append({"role": "user", "content": h['query']})
        messages.append({"role": "assistant", "content": h['sql']})
    messages.append({"role": "user", "content": user_query})

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": "llama-3.3-70b-versatile", "messages": messages, "temperature": 0.1}
    )
    
    res = response.json()
    if "choices" in res:
        sql_out = res['choices'][0]['message']['content'].strip()
        return sql_out.replace("```sql", "").replace("```", "").strip()
    return "I cannot answer this with the available data."

# ==========================================
# --- 6. UI: HEADER & DOWNLOAD ---
# ==========================================
st.title("🏦 Risk Data Agent")

col_head, col_dl = st.columns([0.8, 0.2])
with col_head:
    user_input = st.text_input("Query Bank Data (Shared Memory):", placeholder="e.g. Which clients have an OSUC > 5000?")

with col_dl:
    try:
        with open("additional_data.xlsx", "rb") as f:
            st.download_button(
                label="Download DB",
                data=f,
                file_name="bank_risk_database.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                icon="💾",
                help="Download the underlying Excel database."
            )
    except FileNotFoundError:
        pass

# ==========================================
# --- 7. MAIN EXECUTION ---
# ==========================================
init_global_history()
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
                st.session_state.df = pd.read_sql(final_sql, conn)
                save_to_global_history(user_input, final_sql)
                st.rerun()
            except Exception as e: st.error(f"SQL Error: {e}")

    with col_r:
        st.markdown("### 📊 Results")
        if "df" in st.session_state:
            st.dataframe(st.session_state.df, use_container_width=True)

# ==========================================
# --- 8. TECHNICAL RATIONALE ---
# ==========================================
st.divider()
st.subheader("🛠️ Technical Architecture & Rationale")
t1, t2, t3, t4 = st.columns(4)
with t1:
    st.markdown("#### 🧠 Llama-3.3-70B")
    st.caption("**SQL Engineering**")
    st.write("Ensures complex JOIN accuracy and strict logic sequencing at sub-second speeds.")
with t2:
    st.markdown("#### 🔍 MiniLM Semantic Retrieval")
    st.caption("**Edge-Optimized Embeddings**")
    st.write("Provides highly accurate, lightweight vector mapping to connect user questions to data rules.")
with t3:
    st.markdown("#### 📑 Ontology & Schema")
    st.caption("**The Guardrails**")
    st.write("The JSON-LD enforces mandatory join logic and business jargon, while the Markdown provides the technical source of truth.")
with t4:
    st.markdown("#### 🏗️ Delta Lake Memory")
    st.caption("**Contextual Persistence**")
    st.write("Stores history in Databricks so the team shares a single 'Sliding Window' of context across users.")

with st.sidebar:
    st.header("🌐 Past User queries")
    for h in history: 
        loc = h.get('user_location') or "Unknown Location"
        st.info(f"📍 {loc}\n🗨️ {h['query']}")

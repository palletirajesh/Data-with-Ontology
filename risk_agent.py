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
# --- 2. THE LIBRARIAN (NOMIC EMBEDDINGS) ---
# ==========================================
@st.cache_resource
def load_nomic_model():
    """Nomic v1.5 handles synonyms like 'Client' vs 'Customer' automatically."""
    return SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)

embedder = load_nomic_model()

@st.cache_resource
def get_db_connection():
    return sql.connect(**DB_CONFIG)

conn = get_db_connection()

# ==========================================
# --- 3. GLOBAL SLIDING WINDOW (MEMORY) ---
# ==========================================
def init_global_history():
    """Verifies that the existing history table is accessible."""
    try:
        with conn.cursor() as cursor:
            # We just do a simple check to ensure the table is there
            cursor.execute("SELECT 1 FROM gen_ai_bank.query_history LIMIT 1")
    except Exception as e:
        st.error("⚠️ Global History Table not found in Databricks.")
        st.info("Please ensure 'gen_ai_bank.query_history' exists in your workspace.")
        # We don't raise the error so the app can still run without history

def save_to_global_history(query, sql_code):
    with conn.cursor() as cursor:
        cursor.execute("INSERT INTO gen_ai_bank.query_history (user_query, generated_sql) VALUES (?, ?)", [query, sql_code])

def load_global_history(limit=10):
    try:
        return pd.read_sql(f"SELECT user_query as query, generated_sql as sql FROM gen_ai_bank.query_history ORDER BY created_at DESC LIMIT {limit}", conn).to_dict('records')
    except: return []

# ==========================================
# --- 4. UNIFIED CONTEXT (ZERO MAINTENANCE) ---
# ==========================================
@st.cache_data
def get_unified_knowledge():
    """Unifies Schema and Join Rules. Nomic will handle the synonyms."""
    knowledge_chunks = []
    
    # 1. Load technical schema
    with open("database_schema.md", "r") as f:
        content = f.read()
        knowledge_chunks.extend(["TABLE_STRUCT: " + t for t in content.split("### TABLE:")[1:]])
    
    # 2. Load Join Rules from JSONLD (The logic the AI can't guess)
    g = rdflib.Graph().parse("knowledge_base.jsonld", format="json-ld")
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
            
    return knowledge_chunks

def get_semantic_context(query, knowledge_base, top_k=5):
    """Finds the most relevant tables/rules using Nomic vectors"""
    query_emb = embedder.encode(f"search_query: {query}", convert_to_tensor=True)
    kb_embs = embedder.encode(knowledge_base, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, kb_embs, top_k=top_k)
    return "\n\n".join([knowledge_base[hit['corpus_id']] for hit in hits[0]])

# ==========================================
# --- 5. THE SQL ENGINEER (GROQ LLAMA-3.3) ---
# ==========================================
def generate_sql(user_query, context, history):
    system_prompt = f"""You are a Databricks SQL Expert. Use this Context:
    {context}
    
    STRICT RULES:
    1. ONLY use Tables/Columns in the Context.
    2. Use MANDATORY JOINs exactly. Sequence tables logically.
    3. Output ONLY raw SQL. No markdown.
    4. If data is missing, say: 'I cannot answer this with the available data.'
    5. Start FROM 'dim_customer' for multi-table joins.
    6. For string filters, use: UPPER(column) = UPPER('value').
    7. TRANSLATION: Map terms like 'Amazon' to card_partner = 'AMAZON'.
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-10:]: # Adding global sliding window context
        messages.append({"role": "user", "content": h['query']})
        messages.append({"role": "assistant", "content": h['sql']})
    messages.append({"role": "user", "content": user_query})

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": "llama-3.3-70b-versatile", "messages": messages, "temperature": 0.1}
    )
    
    res_json = response.json()
    if "choices" in res_json:
        sql_out = res_json['choices'][0]['message']['content'].strip()
        return sql_out.replace("```sql", "").replace("```", "").strip()
    return "I cannot answer this with the available data."

# ==========================================
# --- 6. MAIN UI ---
# ==========================================
st.title("🏦 Risk Data Agent")
init_global_history()
kb = get_unified_knowledge()
history = load_global_history()

user_input = st.text_input("Query Bank Data (Shared Memory):", placeholder="e.g. Clients with FICO > 700")

if user_input:
    context = get_semantic_context(user_input, kb)
    generated_sql = generate_sql(user_input, context, history)
    
    col_l, col_r = st.columns([4, 6])
    with col_l:
        st.markdown("### 📝 SQL Draft")
        final_sql = st.text_area("SQL:", value=generated_sql, height=250)
        if st.button("▶️ Run & Log Globally", type="primary"):
            try:
                st.session_state.df = pd.read_sql(final_sql, conn)
                save_to_global_history(user_input, final_sql)
                st.rerun()
            except Exception as e: st.error(f"SQL Error: {e}")

    with col_r:
        st.markdown("### 📊 Results")
        if "df" in st.session_state:
            st.dataframe(st.session_state.df, use_container_width=True)

with st.sidebar:
    st.header("🌐 Global History")
    for h in history: st.caption(f"🗨️ {h['query']}")

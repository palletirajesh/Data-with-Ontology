import streamlit as st
import pandas as pd
import requests
import json
import rdflib
from databricks import sql

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
# --- 2. CONNECTORS & EMBEDDINGS ---
# ==========================================
@st.cache_resource
def get_db_connection():
    return sql.connect(**DB_CONFIG)

conn = get_db_connection()

def get_databricks_embeddings(text):
    """Calls Databricks Foundation API for semantic vectors"""
    url = f"https://{DB_CONFIG['server_hostname']}/serving-endpoints/databricks-gte-large-en/invocations"
    headers = {"Authorization": f"Bearer {DB_CONFIG['access_token']}", "Content-Type": "application/json"}
    # The API expects a list of inputs
    response = requests.post(url, headers=headers, json={"input": [text]})
    if response.status_code == 200:
        return response.json()['data'][0]['embedding']
    raise Exception(f"Databricks Embedding Error: {response.text}")

# ==========================================
# --- 3. GLOBAL SLIDING WINDOW (DATABRICKS) ---
# ==========================================
def init_global_history():
    """Initializes the shared history table in Databricks"""
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gen_ai_bank.query_history (
                query_id LONG GENERATED ALWAYS AS IDENTITY,
                user_query STRING,
                generated_sql STRING,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

def save_to_global_history(query, sql_code):
    with conn.cursor() as cursor:
        cursor.execute("INSERT INTO gen_ai_bank.query_history (user_query, generated_sql) VALUES (?, ?)", [query, sql_code])

def load_global_history(limit=10):
    try:
        return pd.read_sql(f"SELECT user_query as query, generated_sql as sql FROM gen_ai_bank.query_history ORDER BY created_at DESC LIMIT {limit}", conn).to_dict('records')
    except: return []

# ==========================================
# --- 4. UNIFIED CONTEXT (NO REDUNDANCY) ---
# ==========================================
@st.cache_data
def load_unified_semantic_base():
    """Unifies Schema and Ontology into a single searchable list of facts"""
    semantic_base = []
    
    # 1. Pull Table Structures from MD
    with open("database_schema.md", "r") as f:
        content = f.read()
        table_chunks = ["### TABLE:" + t for t in content.split("### TABLE:")[1:]]
        semantic_base.extend(table_chunks)
    
    # 2. Pull Business Meanings from JSONLD
    g = rdflib.Graph().parse("knowledge_base.jsonld", format="json-ld")
    # Fetch Jargon -> Column mappings
    for s, p, o in g.triples((None, rdflib.URIRef("http://gen_ai_bank.com/ontology#businessJargon"), None)):
        for _, _, col in g.triples((s, rdflib.URIRef("http://gen_ai_bank.com/ontology#mapsToColumn"), None)):
             semantic_base.append(f"BUSINESS RULE: '{o}' refers to database column '{col}'")
             
    return semantic_base

# ==========================================
# --- 5. THE AGENT BRAIN ---
# ==========================================
def generate_sql(user_query, context, global_history):
    messages = [
        {"role": "system", "content": f"""You are a Databricks SQL Expert for a bank. 
        Write raw SQL using this Context:
        {context}
        
        Rules:
        1. ONLY use the Tables and Columns explicitly provided in the Context.
        2. If JOIN rules are provided, use them exactly.
        3. For strings, use UPPER case: UPPER(column) = UPPER('value').
        4. If data is missing, say: 'I cannot answer this with the available data.'
        5. Output ONLY raw SQL. No markdown. No explanations.
        """}
    ]
    
    # Add Sliding Window (Last 4 queries across all users)
    for h in global_history[-10:]:
        messages.append({"role": "user", "content": h['query']})
        messages.append({"role": "assistant", "content": h['sql']})
    
    messages.append({"role": "user", "content": user_query})

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": "llama-3.3-70b-versatile", "messages": messages, "temperature": 0.1}
    )
    
    if response.status_code == 200:
        raw = response.json()['choices'][0]['message']['content'].strip()
        return raw.replace("```sql", "").replace("```", "").strip()
    return "Error generating SQL."

# ==========================================
# --- 6. USER INTERFACE ---
# ==========================================
st.title("🏦 Risk Data Agent")
init_global_history()
semantic_base = load_unified_semantic_base()
global_history = load_global_history()

query = st.text_input("Ask about bank risk (Shared memory across users):", placeholder="e.g. List Amazon customers with high debt")

if query:
    # Combined Retrieval logic: Finding nearest neighbors in both schema and ontology
    # For now, we pass the semantic base as context; in a large DB, you'd vectorize this list
    context_payload = "\n\n".join(semantic_base) 
    
    sql_result = generate_sql(query, context_payload, global_history)
    
    col_l, col_r = st.columns([4, 6])
    with col_l:
        st.markdown("### 📝 SQL Draft")
        final_sql = st.text_area("Review/Edit:", value=sql_result, height=200)
        if st.button("▶️ Run & Log Globally", type="primary"):
            try:
                st.session_state.df = pd.read_sql(final_sql, conn)
                save_to_global_history(query, final_sql)
                st.rerun()
            except Exception as e:
                st.error(f"SQL Error: {e}")

    with col_r:
        st.markdown("### 📊 Data")
        if "df" in st.session_state:
            st.dataframe(st.session_state.df, use_container_width=True)

with st.sidebar:
    st.header("🌐 Global Brain")
    st.caption("Last 10 queries by all users:")
    for h in global_history:
        st.info(f"Q: {h['query']}")

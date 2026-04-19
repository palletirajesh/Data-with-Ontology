import streamlit as st
import pandas as pd
import requests
import json
import rdflib
from databricks import sql

# ==========================================
# --- 1. CONFIGURATION & SECRETS ---
# ==========================================
st.set_page_config(page_title="GenAI Bank Agent", page_icon="🏦", layout="wide")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
DB_CONFIG = {
    "server_hostname": st.secrets["DB_SERVER_HOSTNAME"], 
    "http_path": st.secrets["DB_HTTP_PATH"], 
    "access_token": st.secrets["DB_ACCESS_TOKEN"]
}

# ==========================================
# --- 2. CORE CONNECTORS & LOADERS ---
# ==========================================
@st.cache_resource
def get_db_connection():
    return sql.connect(**DB_CONFIG)

conn = get_db_connection()

def get_databricks_embeddings(text):
    """Uses Databricks Foundation Model API for embeddings"""
    url = f"https://{DB_CONFIG['server_hostname']}/serving-endpoints/databricks-gte-large-en/invocations"
    headers = {"Authorization": f"Bearer {DB_CONFIG['access_token']}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json={"input": [text]})
    if response.status_code == 200:
        return response.json()['data'][0]['embedding']
    raise Exception(f"Databricks Embedding Error: {response.text}")

# ==========================================
# --- 3. PERSISTENT GLOBAL SLIDING WINDOW ---
# ==========================================
def init_history_table():
    """Creates global memory table if it doesn't exist"""
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gen_ai_bank.query_history (
                query_id LONG GENERATED ALWAYS AS IDENTITY,
                user_query STRING,
                generated_sql STRING,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

def save_global_history(query, sql_code):
    with conn.cursor() as cursor:
        cursor.execute("INSERT INTO gen_ai_bank.query_history (user_query, generated_sql) VALUES (?, ?)", [query, sql_code])

def load_global_history(limit=10):
    try:
        return pd.read_sql(f"SELECT user_query as query, generated_sql as sql FROM gen_ai_bank.query_history ORDER BY created_at DESC LIMIT {limit}", conn).to_dict('records')
    except: return []

# ==========================================
# --- 4. UNIFIED CONTEXT ENGINE ---
# ==========================================
@st.cache_data
def get_unified_context():
    """Unifies Ontology (JSONLD) and Schema (MD) to avoid redundancy"""
    unified_lines = []
    
    # Process Markdown Schema
    with open("database_schema.md", "r") as f:
        schema_content = f.read()
        table_chunks = ["### TABLE:" + t for t in schema_content.split("### TABLE:")[1:]]
        unified_lines.extend(table_chunks)
    
    # Process JSONLD Ontology for specific Join Rules
    g = rdflib.Graph().parse("knowledge_base.jsonld", format="json-ld")
    # Adding join logic explicitly from ontology as business rules
    query = """
    PREFIX bank: <http://gen_ai_bank.com/ontology#>
    SELECT ?tableLabel ?targetTableLabel ?sKey ?tKey
    WHERE {
        ?table a bank:Table ;
               rdfs:label ?tableLabel ;
               bank:joinsWith ?join .
        ?join bank:targetTable ?targetTable ;
              bank:sourceKey ?sKey ;
              bank:targetKey ?tKey .
        ?targetTable rdfs:label ?targetTableLabel .
    }
    """
    for row in g.query(query):
        unified_lines.append(f"JOIN RULE: {row.tableLabel} joins with {row.targetTableLabel} ON {row.sKey} = {row.tKey}")
            
    return unified_lines

def semantic_retrieval(query, unified_lines, top_k=5):
    """
    Retrieves the most relevant unified context chunks.
    Note: For a production scale, replace this list-search with Databricks Vector Search.
    """
    # Simple semantic similarity logic would go here using get_databricks_embeddings
    return "\n\n".join(unified_lines[:top_k])

# ==========================================
# --- 5. LLM LOGIC & UI ---
# ==========================================
def generate_sql(user_query, context, history):
    messages = [
        {"role": "system", "content": f"You are a Databricks SQL Expert. Use this context: {context}"}
    ]
    
    # Global Sliding Window Context
    if history:
        messages.append({"role": "system", "content": "Refer to recent global queries if contextually relevant."})
        for h in history[-4:]: # Use last 4 for efficiency
            messages.append({"role": "user", "content": h['query']})
            messages.append({"role": "assistant", "content": h['sql']})

    messages.append({"role": "user", "content": f"New Query: {user_query}"})
    
    # Rules merged into prompt for optimization
    prompt_rules = """
    Rules:
    1. ONLY use provided Tables and Columns.
    2. Use JOIN RULES exactly as written.
    3. Output raw SQL only. No markdown.
    4. Start FROM Bridge Tables like dim_customer if necessary.
    5. Translate jargon (e.g., 'Amazon' -> data_unit = 'AMAZON').
    6. For string datatypes, use UPPER case comparison: UPPER(column) = UPPER('value').
    """
    messages[0]["content"] += prompt_rules

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": "llama-3.3-70b-versatile", "messages": messages, "temperature": 0.1}
    )
    
    if response.status_code == 200:
        raw_output = response.json()['choices'][0]['message']['content'].strip()
        return raw_output.replace("```sql", "").replace("```", "").strip()
    return f"Error: {response.text}"

# --- Main App ---
st.title("🏦 Risk Data Agent")
init_history_table()
unified_context = get_unified_context()
global_history = load_global_history()

user_query = st.text_input("Analyze customer risk:", placeholder="e.g. Amazon customers with high debt")

if user_query:
    with st.status("🧠 Processing across instances...") as status:
        context = semantic_retrieval(user_query, unified_context)
        sql_query = generate_sql(user_query, context, global_history)
        status.update(label="✅ Ready", state="complete")

    col_left, col_right = st.columns([4, 6])
    with col_left:
        st.markdown("### 📝 SQL Code")
        edited_sql = st.text_area("Review Code:", value=sql_query, height=200)
        if st.button("▶️ Execute & Log Globally"):
            st.session_state.df = pd.read_sql(edited_sql, conn)
            save_global_history(user_query, edited_sql)
            st.rerun()

    with col_right:
        if "df" in st.session_state:
            st.dataframe(st.session_state.df, use_container_width=True)

with st.sidebar:
    st.header("🌐 Global Memory")
    for h in global_history:
        st.caption(f"🔹 {h['query']}")

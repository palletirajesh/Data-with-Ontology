import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import requests
import rdflib
import duckdb
from sentence_transformers import SentenceTransformer, util

# ==========================================
# --- 1. CONFIGURATION ---
# ==========================================
st.set_page_config(page_title="Risk Data Agent", page_icon="🏦", layout="wide")
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ==========================================
# --- 2. DUCKDB ENGINE ---
# ==========================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedding_model()

@st.cache_resource
def get_db_connection():
    try:
        conn = duckdb.connect(database='bank_data.db', read_only=False)
        # Register Parquet files
        tables = ['dim_card_association', 'fact_credit_bureau', 'fact_card_ledger', 'dim_customer']
        for table in tables:
            conn.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM '{table}.parquet'")
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_query TEXT, generated_sql TEXT, user_location TEXT
            )
        """)
        return conn
    except Exception as e:
        st.error(f"❌ DB Error: {e}")
        return None

conn = get_db_connection()

# ==========================================
# --- 3. CONTEXT ENGINE (FIXED) ---
# ==========================================
@st.cache_data
def get_unified_knowledge():
    knowledge_chunks = []
    
    # 1. Technical Schema
    try:
        with open("database_schema.md", "r") as f:
            chunks = f.read().split("### TABLE:")
            for c in chunks[1:]:
                knowledge_chunks.append(f"TABLE_SCHEMA: {c.strip()}")
    except: pass
    
    # 2. Business Jargon & Joins from JSON-LD
    try:
        g = rdflib.Graph().parse("knowledge_base.jsonld", format="json-ld")
        # Simplified query to grab everything relevant
        q = """
        SELECT ?label ?jargon WHERE {
            ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label .
            OPTIONAL { ?s <http://com/ontology#businessJargon> ?jargon }
        }"""
        for row in g.query(q):
            if row.jargon:
                knowledge_chunks.append(f"JARGON: '{row.jargon}' refers to column/concept '{row.label}'")
    except: pass
    
    return knowledge_chunks

def get_semantic_context(query, knowledge_base):
    query_emb = embedder.encode(f"search_query: {query}", convert_to_tensor=True)
    kb_embs = embedder.encode(knowledge_base, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, kb_embs, top_k=8)
    return "\n\n".join([knowledge_base[hit['corpus_id']] for hit in hits[0]])

# ==========================================
# --- 4. THE SQL ENGINEER ---
# ==========================================
def generate_sql(user_query, context):
    system_prompt = f"""You are a DuckDB SQL Expert. 
    Context from Metadata:
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
    
    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}],
            "temperature": 0
        }
    ).json()
    
    if "choices" in res:
        sql = res['choices'][0]['message']['content'].strip()
        return sql.replace("```sql", "").replace("```", "").strip()
    return "I cannot answer this with available data."

# ==========================================
# --- 5. UI ---
# ==========================================
st.title("🏦 Risk Data Agent")
user_input = st.text_input("Query:", placeholder="e.g. List clients with Amazon cards")

if user_input:
    kb = get_unified_knowledge()
    context = get_semantic_context(user_input, kb)
    
    # DEBUG SIDEBAR
    with st.sidebar:
        st.subheader("🧠 AI Context")
        st.write(context)

    generated_sql = generate_sql(user_input, context)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.code(generated_sql, language="sql")
        if st.button("Run Query"):
            try:
                df = conn.execute(generated_sql).df()
                st.session_state.last_df = df
            except Exception as e: st.error(e)
            
    with col2:
        if "last_df" in st.session_state:
            st.dataframe(st.session_state.last_df)

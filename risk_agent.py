import os
import streamlit as st
import pandas as pd
import requests
import rdflib
import duckdb

# ==========================================
# --- 1. CONFIGURATION ---
# ==========================================
st.set_page_config(page_title="Risk Data Agent", page_icon="🏦", layout="wide")
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ==========================================
# --- 2. DUCKDB ENGINE (ZERO COST) ---
# ==========================================
@st.cache_resource
def get_db_connection():
    """Maps your 4 Parquet files to SQL tables."""
    try:
        conn = duckdb.connect(database='bank_data.db', read_only=False)
        tables = ['dim_card_association', 'fact_credit_bureau', 'fact_card_ledger', 'dim_customer']
        for table in tables:
            conn.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM '{table}.parquet'")
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_query TEXT, generated_sql TEXT
            )
        """)
        return conn
    except Exception as e:
        st.error(f"❌ Database Engine Error: {e}")
        return None

conn = get_db_connection()

# ==========================================
# --- 3. CONTEXT ENGINE (KEYWORD MATCH) ---
# ==========================================
@st.cache_data
def get_unified_knowledge():
    """Loads schema and jargon without needing heavy AI models."""
    knowledge = []
    try:
        with open("database_schema.md", "r") as f:
            knowledge.append(f.read())
    except: pass
    
    try:
        g = rdflib.Graph().parse("knowledge_base.jsonld", format="json-ld")
        q = "SELECT ?label ?jargon WHERE { ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label . OPTIONAL { ?s <http://com/ontology#businessJargon> ?jargon } }"
        for row in g.query(q):
            if row.jargon:
                knowledge.append(f"Mapping: '{row.jargon}' -> Column '{row.label}'")
    except: pass
    return "\n".join(knowledge)

# ==========================================
# --- 4. THE SQL ENGINEER (GROQ) ---
# ==========================================
def generate_sql(user_query, metadata):
    system_prompt = f"""You are a DuckDB Expert. Use this metadata:
    {metadata}
    
    RULES:
    1. ONLY use Tables/Columns in Context.
    2. Use MANDATORY JOINs exactly. Sequence tables logically.
    3. Output ONLY raw SQL code. No markdown or explanations.
    4. If data is missing, output EXACTLY: "I cannot answer this with the available data."
    5. Always begin with a SELECT clause. For multi-table joins, the main FROM table MUST be 'gen_ai_bank.dim_customer'.
    6. For string filters, use: UPPER(column) = UPPER('value').
    7. TRANSLATION: Apply BUSINESS TRANSLATION RULES strictly to map user jargon to correct columns.
    8. GRANULARITY: Unless the user explicitly uses words like 'count', 'total', or 'how many', ALWAYS return a detailed list of records (SELECT *) rather than a summary or count.
    """
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                "temperature": 0
            }
        ).json()
        sql = response['choices'][0]['message']['content'].strip()
        return sql.replace("```sql", "").replace("```", "").strip()
    except:
        return "SELECT * FROM dim_customer LIMIT 5;"

# ==========================================
# --- 5. UI & EXECUTION ---
# ==========================================
st.title("🏦 Risk Data Agent")
user_input = st.text_input("Ask a question about customers or cards:")

if user_input:
    metadata = get_unified_knowledge()
    generated_sql = generate_sql(user_input, metadata)
    
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.subheader("📝 SQL Code")
        final_sql = st.text_area("Edit SQL if needed:", value=generated_sql, height=150)
        if st.button("Run Query"):
            try:
                df = conn.execute(final_sql).df()
                st.session_state.df = df
                conn.execute("INSERT INTO query_history (user_query, generated_sql) VALUES (?, ?)", [user_input, final_sql])
            except Exception as e:
                st.error(f"SQL Error: {e}")

    with col_r:
        st.subheader("📊 Results")
        if "df" in st.session_state:
            st.dataframe(st.session_state.df)

# History Sidebar
with st.sidebar:
    st.header("🕒 Recent Queries")
    try:
        history = conn.execute("SELECT user_query FROM query_history ORDER BY created_at DESC LIMIT 5").df()
        for q in history['user_query']:
            st.write(f"- {q}")
    except: pass

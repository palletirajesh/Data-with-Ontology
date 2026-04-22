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
    """Initializes DuckDB and maps your Parquet files to SQL tables."""
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
# --- 3. CONTEXT ENGINE (LIGHTWEIGHT) ---
# ==========================================
@st.cache_data
def get_unified_knowledge():
    """Loads schema and jargon directly from your files."""
    knowledge = []
    try:
        with open("database_schema.md", "r") as f:
            knowledge.append(f"DATABASE_SCHEMA:\n{f.read()}")
    except: pass
    
    try:
        g = rdflib.Graph().parse("knowledge_base.jsonld", format="json-ld")
        q = "SELECT ?label ?jargon WHERE { ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label . OPTIONAL { ?s <http://com/ontology#businessJargon> ?jargon } }"
        for row in g.query(q):
            if row.jargon:
                knowledge.append(f"BUSINESS_RULE: '{row.jargon}' refers to column '{row.label}'")
    except: pass
    return "\n\n".join(knowledge)

# ==========================================
# --- 4. THE SQL ENGINEER (GROQ) ---
# ==========================================
def generate_sql(user_query, metadata):
    system_prompt = f"""You are a DuckDB SQL Expert. 
    Use this metadata to build queries:
    {metadata}
    
    STRICT RULES:
    1. ONLY use tables: dim_customer, dim_card_association, fact_card_ledger, fact_credit_bureau.
    2. Primary table is dim_customer. JOIN others as needed.
    3. Output ONLY raw SQL. No markdown, no explanations.
    4. For 'Amazon' cards, JOIN dim_customer to dim_card_association ON card_id and filter card_product_name.
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
        return response['choices'][0]['message']['content'].strip().replace("```sql", "").replace("```", "").strip()
    except Exception as e:
        return f"-- Error generating SQL: {e}"

# ==========================================
# --- 5. UI ---
# ==========================================
st.title("🏦 Risk Data Agent")
st.info("System optimized for Python 3.14 (Lightweight Mode)")

user_input = st.text_input("Ask a question about your risk data:")

if user_input:
    metadata = get_unified_knowledge()
    generated_sql = generate_sql(user_input, metadata)
    
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.subheader("📝 SQL Logic")
        final_sql = st.text_area("SQL:", value=generated_sql, height=200)
        if st.button("Run Query", type="primary"):
            try:
                res_df = conn.execute(final_sql).df()
                st.session_state.df = res_df
                conn.execute("INSERT INTO query_history (user_query, generated_sql) VALUES (?, ?)", [user_input, final_sql])
            except Exception as e: st.error(f"SQL Error: {e}")

    with col_r:
        st.subheader("📊 Results")
        if "df" in st.session_state:
            st.dataframe(st.session_state.df)

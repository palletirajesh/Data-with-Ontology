import streamlit as st
import pandas as pd
import requests
import time
import json
import rdflib
from rdflib import URIRef, Literal, Namespace
from databricks import sql
from sentence_transformers import SentenceTransformer, util

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
# --- 2. CORE CONNECTIONS & LOADERS ---
# ==========================================
@st.cache_resource
def get_db_connection():
    return sql.connect(**DB_CONFIG)

conn = get_db_connection()

@st.cache_resource
def load_vector_model():
    # tiny but powerful model for finding semantic similarity
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_ontology_as_list():
    """Converts Graph Triples into searchable semantic strings"""
    g = rdflib.Graph()
    try:
        g.parse("knowledge_base.jsonld", format="json-ld")
        rules = []
        EX = Namespace("http://example.org/ontology/")
        # Extract every jargon mapping as a descriptive sentence for the vector model
        for s, p, o in g.triples((None, EX.businessJargon, None)):
            jargon = str(o)
            for _, _, col in g.triples((s, EX.mapsToColumn, None)):
                rules.append(f"Business Jargon: '{jargon}' maps to Database Column: {str(col)}")
        return rules
    except Exception as e:
        st.error(f"Failed to load Ontology: {e}")
        return []

@st.cache_resource
def load_database_schema():
    try:
        with open("database_schema.md", "r") as file:
            return file.read()
    except FileNotFoundError:
        return ""

# ==========================================
# --- 3. THE SEMANTIC SEARCH ENGINE ---
# ==========================================
def semantic_context_retrieval(user_query, ontology_rules, schema_text, top_k_rules=3, top_k_tables=3):
    """
    Finds the nearest business rules (Ontology) AND nearest tables (Schema) 
    using vector similarity instead of exact matching.
    """
    model = load_vector_model()
    query_emb = model.encode(user_query)

    # --- Search Layer 1: Semantic Jargon/Ontology ---
    matched_ontology = "Relevant Business Rules (Semantic Match):\n"
    if ontology_rules:
        rule_embs = model.encode(ontology_rules)
        rule_hits = util.semantic_search(query_emb, rule_embs, top_k=top_k_rules)[0]
        for hit in rule_hits:
            if hit['score'] > 0.40: # Threshold to ensure relevance
                matched_ontology += f"- {ontology_rules[hit['corpus_id']]}\n"

    # --- Search Layer 2: Semantic Schema ---
    table_chunks = ["### TABLE:" + t for t in schema_text.split("### TABLE:")[1:]]
    table_embs = model.encode(table_chunks)
    table_hits = util.semantic_search(query_emb, table_embs, top_k=top_k_tables)[0]
    
    matched_schema = "Relevant Database Tables:\n"
    for hit in table_hits:
        matched_schema += f"{table_chunks[hit['corpus_id']]}\n\n"

    return matched_ontology + "\n" + matched_schema

# ==========================================
# --- 4. LLM & UI LOGIC ---
# ==========================================
def generate_sql(user_query, context):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    prompt = f"""You are an expert Databricks SQL data engineer.
    Write a SQL query based on the Semantic Context provided.
    
    [Context]:
    {context}

    [User Question]:
    {user_query}

    Rules:
    1. ONLY use the Tables and Columns explicitly provided in the ONTOLOGY SCHEMA MATCHES above.
    2. If MANDATORY JOIN RULES are provided, you MUST use them exactly as written. If no join rules are provided but multiple tables exist, join them logically.
    3. Output ONLY the raw SQL code. No markdown, no explanations.
    4. SEQUENCE YOUR JOINS LOGICALLY. You cannot reference a table alias in an ON clause until that table has been introduced. If dim_customer is provided as a Bridge Table, start your FROM clause there.
    5. If the user asks for a column or metric that cannot be calculated from the given columns, output exactly: "I cannot answer this with the available data."
    6. TRANSLATION: Do NOT copy the user's words into the SQL. If the user asks for "payment delay", you MUST translate that to the exact column 'days_past_due'.
    7. Output ONLY the raw SQL code. No markdown, no explanations.
    8. For the string datatype, convert any case to Upper Case like upper(data_unit) = upper("Amazon')
    """
    
    payload = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        raw_output = response.json()['choices'][0]['message']['content'].strip()
        # Clean up accidental markdown blocks
        return raw_output.replace("```sql", "").replace("```", "").strip()
    return f"Error: {response.text}"

# ==========================================
# --- 5. UI & SESSION STATE ---
# ==========================================
if "df" not in st.session_state: st.session_state.df = None
if "sql" not in st.session_state: st.session_state.sql = ""
if "context" not in st.session_state: st.session_state.context = ""

st.title("🏦 Hybrid Semantic Bank Agent")

# Load external data
ontology_rules = load_ontology_as_list()
db_schema = load_database_schema()

user_query = st.text_input("Ask a question about customer risk:", placeholder="e.g., customers with amazon account and due > 5000")

if user_query:
    with st.status("🧠 Searching Business Rules & Schema Semantically...") as status:
        # Perform Vector Search for both the rules and the tables
        st.session_state.context = semantic_context_retrieval(user_query, ontology_rules, db_schema)
        st.session_state.sql = generate_sql(user_query, st.session_state.context)
        status.update(label="✅ Strategy: Semantic RAG complete", state="complete", expanded=False)

    # Layout: Left column for SQL, Right for Data
    col_left, col_right = st.columns([4, 6])
    
    with col_left:
        st.markdown("### 📝 SQL Code")
        edited_sql = st.text_area("Review/Edit SQL:", value=st.session_state.sql, height=200)
        
        if st.button("▶️ Execute Query", type="primary", use_container_width=True):
            try:
                st.session_state.df = pd.read_sql(edited_sql, conn)
            except Exception as e:
                st.error(f"SQL Error: {e}")

        with st.expander("🔍 View Semantic Context Payload"):
            st.info(st.session_state.context)

    with col_right:
        st.markdown("### 📊 Query Results")
        if st.session_state.df is not None:
            st.dataframe(st.session_state.df, use_container_width=True)
        else:
            st.info("Results will appear here once query is executed.")

# Sidebar Download
with st.sidebar:
    st.header("📂 Data Resources")
    try:
        with open("additional_data.xlsx", "rb") as f:
            st.download_button("📥 Download Sample Data", f, "Bank_Data.xlsx", type="secondary")
    except: pass

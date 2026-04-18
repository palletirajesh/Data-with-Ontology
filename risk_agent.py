import streamlit as st
import pandas as pd
import requests
import time
import rdflib
import os
from databricks import sql

# ==========================================
# --- 1. CONFIGURATION & SECRETS ---
# ==========================================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

DB_CONFIG = {
    "server_hostname": st.secrets["DB_SERVER_HOSTNAME"], 
    "http_path": st.secrets["DB_HTTP_PATH"], 
    "access_token": st.secrets["DB_ACCESS_TOKEN"]
}

# ==========================================
# --- 2. CORE CONNECTIONS & FILE LOADERS ---
# ==========================================
@st.cache_resource
def get_db_connection():
    return sql.connect(**DB_CONFIG)

conn = get_db_connection()

@st.cache_resource
def load_ontology():
    """Loads the RDF/OWL Graph into memory"""
    g = rdflib.Graph()
    try:
        g.parse("knowledge_base.jsonld", format="json-ld")
    except Exception as e:
        st.error(f"Failed to load Ontology: {e}")
    return g

@st.cache_resource
def load_database_schema():
    """Loads the database schema from the external markdown file."""
    try:
        with open("database_schema.md", "r") as file:
            return file.read()
    except FileNotFoundError:
        return "ERROR: database_schema.md file not found."

# ==========================================
# --- 3. THE HYBRID ONTOLOGY ENGINE ---
# ==========================================
def get_ontology_context(user_query, g):
    query_lower = user_query.lower()
    matched_concepts = []
    
    for s, p, o in g.triples((None, rdflib.URIRef("http://example.org/ontology/businessJargon"), None)):
        jargon_term = str(o).lower()
        if jargon_term in query_lower:
            matched_concepts.append(s)

    if not matched_concepts:
        return "No proprietary business rules found in the ontology for this query. Rely strictly on standard SQL knowledge and the provided Database Schema."

    formatted_graph_context = "Proprietary Business Rules Found:\n"
    for concept in matched_concepts:
        for _, _, col in g.triples((concept, rdflib.URIRef("http://example.org/ontology/mapsToColumn"), None)):
            formatted_graph_context += f"- Map '{concept.split('/')[-1]}' to column: {col}\n"
            
    return formatted_graph_context

# ==========================================
# --- 4. THE LLM SQL GENERATOR ---
# ==========================================
def generate_sql(user_query, ontology_context, database_schema):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""You are an expert Databricks SQL data engineer for a bank.
    Your job is to translate the user's natural language question into a valid Databricks SQL query.

    [Database Schema]:
    {database_schema}

    [Ontology Business Rules]:
    {ontology_context}

    [User Question]:
    {user_query}

    Rules:
    1. Only return the raw SQL code. No markdown formatting, no explanations, no backticks.
    2. Ensure the SQL is compatible with Databricks.
    """
    
    payload = {
        "model": "llama-3.3-70b-versatile", 
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1 
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        raise Exception(f"Groq API Error: {response.text}")

# ==========================================
# --- 5. THE STREAMLIT UI ---
# ==========================================
st.set_page_config(page_title="GenAI Bank Agent", page_icon="🏦", layout="wide")
st.title("🏦 GenAI Graph-RAG Bank Agent")

# Load Files
g = load_ontology()
db_schema = load_database_schema()

user_query = st.text_input("Ask a question about customer risk:")

if user_query:
    with st.spinner("🧠 Consulting Ontology & Schema..."):
        
        ontology_context = get_ontology_context(user_query, g)
        
        with st.expander("🔍 See AI Thought Process"):
            st.info(ontology_context)
            
        try:
            # We now pass db_schema into the LLM function!
            generated_sql = generate_sql(user_query, ontology_context, db_schema)
            
            st.success("SQL Generated Successfully!")
            st.code(generated_sql, language="sql")
            
            with st.spinner("⚡ Fetching data from Databricks..."):
                exec_start = time.time()
                df = pd.read_sql(generated_sql, conn)
                
                st.markdown("### 📊 Results")
                st.dataframe(df, use_container_width=True)
                st.caption(f"Data retrieved in {round(time.time() - exec_start, 2)}s")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

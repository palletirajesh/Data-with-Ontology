import streamlit as st
import pandas as pd
import requests
import re
import time
import json
import rdflib
import os
from databricks import sql

# ==========================================
# --- 1. CONFIGURATION & SECRETS ---
# ==========================================
# Make sure your Streamlit Secrets or secrets.toml match these keys exactly!
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

DB_CONFIG = {
    "server_hostname": st.secrets["DB_SERVER_HOSTNAME"], 
    "http_path": st.secrets["DB_HTTP_PATH"], 
    "access_token": st.secrets["DB_ACCESS_TOKEN"]
}

# ==========================================
# --- 2. DATABASE SCHEMA (CRITICAL FOR HYBRID AI) ---
# ==========================================
# The LLM needs this map so it knows what tables exist when the ontology is empty.
# Update this to match your actual Databricks tables!
DATABASE_SCHEMA = """
Table: customers
Columns: customer_id (string), name (string), risk_score (integer), location (string)

Table: transactions
Columns: transaction_id (string), customer_id (string), amount (decimal), delay_days (integer), status (string)
"""

# ==========================================
# --- 3. CORE CONNECTIONS ---
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

# ==========================================
# --- 4. THE HYBRID ONTOLOGY ENGINE ---
# ==========================================
def get_ontology_context(user_query, g):
    """Searches the graph. If it fails, it returns a soft fallback instead of an error."""
    # Convert query to lowercase for simple matching
    query_lower = user_query.lower()
    
    # 1. Extract concepts dynamically based on the graph
    # (Assuming you have businessJargon nodes in your JSON-LD)
    matched_concepts = []
    for s, p, o in g.triples((None, rdflib.URIRef("http://example.org/ontology/businessJargon"), None)):
        jargon_term = str(o).lower()
        if jargon_term in query_lower:
            matched_concepts.append(s)

    # 2. THE HYBRID FALLBACK: If no exact jargon is found, don't crash!
    if not matched_concepts:
        return "No proprietary business rules found in the ontology for this query. Rely strictly on standard SQL knowledge and the provided Database Schema."

    # 3. If concepts ARE found, build the strict context (Multi-Hop)
    formatted_graph_context = "Proprietary Business Rules Found:\n"
    for concept in matched_concepts:
        # Example: Find the table and column mapped to this concept
        for _, _, col in g.triples((concept, rdflib.URIRef("http://example.org/ontology/mapsToColumn"), None)):
            formatted_graph_context += f"- Map '{concept.split('/')[-1]}' to column: {col}\n"
            
    return formatted_graph_context

# ==========================================
# --- 5. THE LLM SQL GENERATOR ---
# ==========================================
def generate_sql(user_query, ontology_context):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""You are an expert Databricks SQL data engineer for a bank.
    Your job is to translate the user's natural language question into a valid Databricks SQL query.

    [Database Schema]:
    {DATABASE_SCHEMA}

    [Ontology Business Rules]:
    {ontology_context}

    [User Question]:
    {user_query}

    Rules:
    1. Only return the raw SQL code. No markdown formatting, no explanations, no backticks.
    2. Ensure the SQL is compatible with Databricks.
    3. If Ontology Business Rules exist, you MUST follow them. If they say "No proprietary rules found", rely entirely on the Database Schema.
    """
    
    payload = {
        "model": "llama3-70b-8192", # Using a highly capable model for SQL
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1 # Low temperature for strict coding
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        raise Exception(f"Groq API Error: {response.text}")

# ==========================================
# --- 6. THE STREAMLIT UI ---
# ==========================================
st.set_page_config(page_title="GenAI Bank Agent", page_icon="🏦", layout="wide")
st.title("🏦 GenAI Graph-RAG Bank Agent")
st.markdown("Ask natural language questions. The AI will use our semantic ontology and database schema to write Databricks SQL.")

# Load Graph
g = load_ontology()

# User Input
user_query = st.text_input("Ask a question about customer risk:", placeholder="e.g., How many customers do we have? OR Show high-risk customers.")

if user_query:
    with st.spinner("🧠 Consulting the Ontology and generating SQL..."):
        
        # 1. Check the Ontology (Hybrid Approach)
        ontology_context = get_ontology_context(user_query, g)
        
        with st.expander("🔍 See AI Thought Process (Ontology Context)"):
            st.info(ontology_context)
            
        try:
            # 2. Generate SQL
            generated_sql = generate_sql(user_query, ontology_context)
            
            st.success("SQL Generated Successfully!")
            st.code(generated_sql, language="sql")
            
            # 3. Execute SQL
            with st.spinner("⚡ Fetching data from Databricks..."):
                exec_start = time.time()
                df = pd.read_sql(generated_sql, conn)
                
                st.markdown("### 📊 Results")
                st.dataframe(df, use_container_width=True)
                st.caption(f"Data retrieved in {round(time.time() - exec_start, 2)}s")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

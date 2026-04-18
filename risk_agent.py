import streamlit as st
import pandas as pd
import requests
import time
import json
import re
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
# --- 2. CORE CONNECTIONS & FILE LOADERS ---
# ==========================================
@st.cache_resource
def get_db_connection():
    return sql.connect(**DB_CONFIG)

conn = get_db_connection()

@st.cache_resource
def load_ontology():
    g = rdflib.Graph()
    try:
        g.parse("knowledge_base.jsonld", format="json-ld")
    except Exception as e:
        st.error(f"Failed to load Ontology: {e}")
    return g

@st.cache_resource
def load_database_schema():
    try:
        with open("database_schema.md", "r") as file:
            return file.read()
    except FileNotFoundError:
        return ""

@st.cache_resource
def load_vector_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# ==========================================
# --- 3. THE ROUTING ENGINES ---
# ==========================================
def get_ontology_context(user_query, g):
    query_lower = user_query.lower()
    matched_concepts = []
    
    for s, p, o in g.triples((None, rdflib.URIRef("http://example.org/ontology/businessJargon"), None)):
        jargon_term = str(o).lower()
        if jargon_term in query_lower:
            matched_concepts.append(s)

    if not matched_concepts:
        return None 

    formatted_context = "Proprietary Jargon Matched:\n"
    for concept in matched_concepts:
        for _, _, col in g.triples((concept, rdflib.URIRef("http://example.org/ontology/mapsToColumn"), None)):
            formatted_context += f"- Map '{concept.split('/')[-1]}' to column: {col}\n"
            
    return formatted_context

# Updated top_k to 4 to handle complex multi-table queries
def vector_search_schema(user_query, schema_text, top_k=4):
    model = load_vector_model()
    chunks = ["### TABLE:" + t for t in schema_text.split("### TABLE:")[1:]]
    if not chunks:
        return "Error parsing schema file."

    query_embedding = model.encode(user_query)
    chunk_embeddings = model.encode(chunks)
    hits = util.semantic_search(query_embedding, chunk_embeddings, top_k=top_k)[0]
    
    retrieved_schema = "Vector Search Retrieved Tables:\n"
    for hit in hits:
        retrieved_schema += chunks[hit['corpus_id']] + "\n\n"
        
    return retrieved_schema

# ==========================================
# --- 4. LLM & LEARNING ENGINES ---
# ==========================================
def generate_sql(user_query, context):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    prompt = f"""You are an expert Databricks SQL data engineer for a bank.
    Write a Databricks SQL query for the user's question based strictly on the provided Context.

    [Context Provided by Search Engine]:
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
        # Clean out any accidental markdown blocks the LLM might have added
        cleaned_sql = re.sub(r"^
http://googleusercontent.com/immersive_entry_chip/0

Make sure your `additional_data.xlsx` file is safely uploaded alongside this in your GitHub repository, sync the changes, and you will be good to go!

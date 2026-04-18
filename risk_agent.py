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

# Uses top_k=4 to handle complex multi-table queries
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
        # Bulletproof cleanup to remove markdown formatting safely
        cleaned_sql = raw_output.replace("```sql", "").replace("```", "").strip()
        return cleaned_sql
    else:
        raise Exception(f"Groq API Error: {response.text}")

def auto_update_ontology(user_query, original_sql, edited_sql):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    prompt = f"""
    The user asked: "{user_query}"
    The AI wrote: {original_sql}
    The User corrected it to: {edited_sql}
    
    Identify the specific business jargon from the user's question, and the exact database column the user mapped it to in their corrected SQL.
    Respond ONLY with a valid JSON object in this exact format:
    {{"jargon": "the phrase", "column": "the_table_column"}}
    """
    
    try:
        response = requests.post(url, headers=headers, json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }).json()
        
        learning = json.loads(response['choices'][0]['message']['content'])
        jargon = learning['jargon'].lower()
        column = learning['column']
        
        g = load_ontology()
        EX = Namespace("http://example.org/ontology/")
        concept_uri = EX[f"concept_{jargon.replace(' ', '_')}"]
        
        g.add((concept_uri, EX.businessJargon, Literal(jargon)))
        g.add((concept_uri, EX.mapsToColumn, Literal(column)))
        g.serialize(destination="knowledge_base.jsonld", format="json-ld")
        
        st.cache_resource.clear() 
        return True, f"🧠 AI Learned! Mapped '{jargon}' to '{column}'."
    except Exception as e:
        return False, str(e)

# ==========================================
# --- 5. UI SESSION STATE ---
# ==========================================
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "original_sql" not in st.session_state:
    st.session_state.original_sql = ""
if "df" not in st.session_state:
    st.session_state.df = None
if "final_context" not in st.session_state:
    st.session_state.final_context = ""
if "routing_path" not in st.session_state:
    st.session_state.routing_path = ""

# ==========================================
# --- 6. FRONTEND LAYOUT ---
# ==========================================

# --- SIDEBAR: STATIC EXCEL DOWNLOAD ---
with st.sidebar:
    st.header("📂 Explore the Data")
    st.markdown("Download a static snapshot of the database to see what kind of questions you can ask.")
    
    try:
        # Reads the static file directly from your GitHub repo
        with open("additional_data.xlsx", "rb") as file:
            st.download_button(
                label="📥 Download Data.xlsx",
                data=file,
                file_name="Bank_Sample_Data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )
    except FileNotFoundError:
        st.warning("⚠️ 'additional_data.xlsx' not found. Please upload it to your repository.")

# --- MAIN HEADER & INPUT ---
st.title("🏦 GenAI Graph-RAG Bank Agent")

g = load_ontology()
db_schema = load_database_schema()

# Clean, full-width input box
user_query = st.text_input("Ask a question about customer risk:", placeholder="e.g. Show me bad debt accounts.")

# --- HORIZONTAL SPLIT LAYOUT ---
# Left column takes 35% of the screen, Right column takes 65%
col_left, col_spacer, col_right = st.columns([3.5, 0.5, 6])

if user_query and user_query != st.session_state.last_query:
    st.session_state.last_query = user_query
    st.session_state.df = None
    
    with col_left:
        with st.status("🧠 Analyzing your question...", expanded=True) as status:
            ontology_context = get_ontology_context(user_query, g)
            
            if ontology_context:
                st.session_state.routing_path = "Ontology Search"
                st.session_state.final_context = f"{ontology_context}\n\nSchema Fallback:\n{db_schema}"
                status.update(label="✅ Strategy: Ontology Exact Match", state="complete", expanded=False)
            else:
                st.session_state.routing_path = "Vector Search"
                st.session_state.final_context = vector_search_schema(user_query, db_schema)
                status.update(label="✅ Strategy: Vector Semantic Search", state="complete", expanded=False)
                
            try:
                st.session_state.original_sql = generate_sql(user_query, st.session_state.final_context)
            except Exception as e:
                st.error(str(e))
                st.stop()

# --- POPULATE COLUMNS ---
if st.session_state.original_sql:
    
    # LEFT COLUMN: Logic, Code Editing, and Learning
    with col_left:
        st.markdown("### 📝 Code Editor")
        
        edited_sql = st.text_area(
            "Modify the AI's SQL code:", 
            value=st.session_state.original_sql, 
            height=200,
            label_visibility="collapsed"
        )
        
        if st.button("▶️ Execute Query", type="primary", use_container_width=True):
            with st.spinner("Fetching data..."):
                try:
                    st.session_state.df = pd.read_sql(edited_sql, conn)
                except Exception as e:
                    st.error(f"❌ SQL Execution Error: {str(e)}")
                    st.session_state.df = None
                    
        # Teacher Agent Block
        if edited_sql.strip() != st.session_state.original_sql.strip():
            st.warning("⚠️ You modified the AI's query.")
            if st.button("🕸️ Teach AI your edits", use_container_width=True):
                with st.spinner("Wiring new semantic edges..."):
                    success, msg = auto_update_ontology(st.session_state.last_query, st.session_state.original_sql, edited_sql)
                    if success:
                        st.success(msg)
                        st.session_state.original_sql = edited_sql
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error(msg)
                        
        with st.expander("🔍 View AI Context Payload"):
            st.code(st.session_state.final_context, language="markdown")

    # RIGHT COLUMN: Data Results
    with col_right:
        st.markdown("### 📊 Query Results")
        if st.session_state.df is not None:
            st.dataframe(st.session_state.df, use_container_width=True, height=500)
        else:
            st.info("👈 Run the query to see the resulting data here.")

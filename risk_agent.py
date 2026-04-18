import streamlit as st
import pandas as pd
import requests
import time
import rdflib
import io
from databricks import sql
from sentence_transformers import SentenceTransformer, util

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

def vector_search_schema(user_query, schema_text, top_k=2):
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
# --- 4. THE LLM SQL GENERATOR ---
# ==========================================
def generate_sql(user_query, context):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
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

# --- SIDEBAR: EXCEL DATA DOWNLOAD ---
with st.sidebar:
    st.header("📂 Explore the Data")
    st.write("Download a snapshot of the database to see what kind of questions you can ask.")
    
    if st.button("Generate Sample Excel File"):
        with st.spinner("Extracting samples from Databricks..."):
            try:
                # Pull 50 rows from the core tables
                df_cust = pd.read_sql("SELECT * FROM gen_ai_bank.dim_customer LIMIT 50", conn)
                df_ledger = pd.read_sql("SELECT * FROM gen_ai_bank.fact_card_ledger LIMIT 50", conn)
                
                # Write to an in-memory Excel file
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_cust.to_excel(writer, sheet_name='Customers', index=False)
                    df_ledger.to_excel(writer, sheet_name='Card_Ledger', index=False)
                
                # Provide the download button
                st.download_button(
                    label="📥 Download Data.xlsx",
                    data=buffer.getvalue(),
                    file_name="additional_data.xlsx",
                    mime="application/vnd.ms-excel",
                    type="primary"
                )
            except Exception as e:
                st.error(f"Failed to fetch sample data: {e}")

# --- MAIN PAGE ---
st.title("🏦 GenAI Graph-RAG Bank Agent")

g = load_ontology()
db_schema = load_database_schema()

user_query = st.text_input("Ask a question about customer risk:", placeholder="e.g. Show me the number of customers, or What is our total ENR?")

if user_query:
    # 1. The Interactive Thought Process Tracker
    with st.status("🧠 Analyzing your question...", expanded=True) as status:
        
        st.write("🔎 **Step 1:** Checking Ontology for strict business jargon...")
        time.sleep(1) # Slight pause for visual effect
        ontology_context = get_ontology_context(user_query, g)
        
        if ontology_context:
            st.success("🎯 **Decision:** Business jargon found! Using Ontology exact match.")
            routing_path = "Ontology Search"
            final_context = f"{ontology_context}\n\nSchema Fallback:\n{db_schema}"
        else:
            st.warning("🧲 **Decision:** No strict jargon found. Switching to Vector Semantic Search...")
            vector_context = vector_search_schema(user_query, db_schema)
            final_context = vector_context
            st.info("✅ Relevant tables mathematically matched and retrieved.")
            routing_path = "Vector Search"
            
        st.write(f"📝 **Step 2:** Sending context to LLM via {routing_path}...")
        
        try:
            generated_sql = generate_sql(user_query, final_context)
            status.update(label=f"✅ Strategy Complete (Routed via {routing_path})", state="complete", expanded=False)
            
            # Show the context to the user so they can inspect it
            with st.expander("View the exact context sent to the LLM"):
                st.code(final_context, language="markdown")
                
        except Exception as e:
            status.update(label="❌ API Error", state="error", expanded=True)
            st.error(str(e))
            st.stop()

    # 2. Execution & Results
    st.success("SQL Generated Successfully!")
    st.code(generated_sql, language="sql")
    
    with st.spinner("⚡ Fetching data from Databricks..."):
        exec_start = time.time()
        try:
            df = pd.read_sql(generated_sql, conn)
            st.markdown("### 📊 Query Results")
            st.dataframe(df, use_container_width=True)
            st.caption(f"Data retrieved in {round(time.time() - exec_start, 2)}s")
        except Exception as e:
            st.error(f"❌ Databricks Execution Error: {str(e)}")

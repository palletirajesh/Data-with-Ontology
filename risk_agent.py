import os
import time
import streamlit as st
import pandas as pd
import requests
import rdflib
from google.cloud import bigquery
from google.oauth2 import service_account
from sentence_transformers import SentenceTransformer, util

# --- 1. CONFIG & STYLE ---
st.set_page_config(page_title="Risk Data Agent", page_icon="🏦", layout="wide")

# CSS for Professional Fonts and Alignment
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .stButton>button { width: 100%; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
BQ_PROJECT = st.secrets["bigquery"]["project_id"]
BQ_DATASET = st.secrets["bigquery"]["dataset_id"]

# --- 2. ENGINES ---
@st.cache_resource
def get_bq_client():
    info = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(info)
    return bigquery.Client(credentials=credentials, project=BQ_PROJECT)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

bq_client = get_bq_client()
embedder = load_embedding_model()

# --- 3. DATA DOWNLOADER ---
def get_full_dataset():
    # Helper to let users download the sample DB for context
    # Note: For production, you might want to pre-save a sample.csv in GitHub
    query = f"SELECT * FROM `{BQ_PROJECT}.{BQ_DATASET}.dim_customer` LIMIT 100"
    return bq_client.query(query).to_dataframe().to_csv(index=False)

# --- 4. CONTEXT & SQL GENERATION ---
def generate_sql(user_query, context, history):
    full_path = f"{BQ_PROJECT}.{BQ_DATASET}"
    
    # Updated Prompt for Intelligent Column Selection
    system_prompt = f"""You are a BigQuery SQL Expert. 
    Context: {context}
    
    STRICT RULES:
    1. ONLY use Tables/Columns in Context.
    2. Table Format: ALWAYS use backticks and the full path: `{full_path}.table_name`
    3. NO PARENTHESES: Never put () after a table name.
    4. Use MANDATORY JOINs exactly. Sequence tables logically.
    5. Output ONLY raw SQL code. No markdown or explanations.
    6. If data is missing, output EXACTLY: "I cannot answer this with the available data."
    7. Always begin with a SELECT clause. For multi-table joins, dim_customer should act a bridge table
    8. EVERY table name in the SQL must be prefixed with: `{full_path}.
    9. Example format: SELECT * FROM `{full_path}.dim_customer` JOIN `{full_path}.dim_card_association`
    10. Tables available: dim_customer, dim_card_association, fact_card_ledger, fact_credit_bureau.
    11. For string filters, use: UPPER(column) = UPPER('value').
    12. TRANSLATION: Apply BUSINESS TRANSLATION RULES strictly to map user jargon to correct columns.
    13. GRANULARITY: Unless the user explicitly uses words like 'count', 'total', or 'how many', ALWAYS return a detailed list of records (SELECT *) rather than a summary or count.
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-2:]:
        messages.append({"role": "user", "content": h['query']})
        messages.append({"role": "assistant", "content": h['sql']})
    messages.append({"role": "user", "content": user_query})

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": "llama-3.3-70b-versatile", "messages": messages, "temperature": 0.1}
    ).json()
    
    return response['choices'][0]['message']['content'].strip().replace("```sql", "").replace("```", "").strip()

# --- 5. SIDEBAR HISTORY ---
with st.sidebar:
    st.title("🕒 Query History")
    # Fetching history from BQ
    try:
        hist_query = f"SELECT user_query FROM `{BQ_PROJECT}.{BQ_DATASET}.query_history` ORDER BY created_at DESC LIMIT 10"
        hist_df = bq_client.query(hist_query).to_dataframe()
        for q in hist_df['user_query']:
            st.caption(f"🗨️ {q}")
            st.divider()
    except:
        st.write("No history found.")

# --- 6. MAIN UI ---
st.title("🏦 Risk Data Intelligence Agent")

# Download Button Section
col_dl, _ = st.columns([1, 4])
with col_dl:
    csv_data = get_full_dataset()
    st.download_button(
        label="📥 Download Data Dictionary (CSV)",
        data=csv_data,
        file_name="bank_data_sample.csv",
        mime="text/csv",
    )

user_input = st.text_input("Analyze your risk data:", placeholder="e.g. List the names and scores of clients with Amazon cards")

# Step 6: Logic to clear results before new execution
if user_input:
    # We use a placeholder to clear the screen
    result_container = st.empty()
    
    # 1. Start Intelligence logic
    kb = get_unified_knowledge() # Assume your existing function
    context = get_semantic_context(user_input, kb)
    generated_sql = generate_sql(user_input, context, [])

    col_sql, col_res = st.columns([4, 6])
    
    with col_sql:
        st.subheader("📝 Generated SQL")
        final_sql = st.text_area("Review Code:", value=generated_sql, height=200)
        
        # Step 5: Execution Status and Timer
        if st.button("▶️ Execute Query"):
            # Clear previous results from session state
            if "df" in st.session_state:
                del st.session_state.df
                
            start_time = time.perf_counter()
            with st.status("🚀 Executing on BigQuery...", expanded=True) as status:
                try:
                    df = bq_client.query(final_sql).to_dataframe()
                    end_time = time.perf_counter()
                    st.session_state.df = df
                    status.update(label=f"✅ Query Successful in {end_time - start_time:.2f}s!", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="❌ Execution Failed", state="error")
                    st.error(e)

    with col_res:
        st.subheader("📊 Data Results")
        if "df" in st.session_state:
            st.dataframe(st.session_state.df, use_container_width=True)

# --- 7. REFINED ARCHITECTURE (Step 4) ---
st.divider()
st.subheader("🏗️ Agent Architecture")

# Professional flow diagram-like layout
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.info("**1. User Input**\nNatural language captured via Streamlit UI.")
with c2:
    st.info("**2. Semantic Mapping**\nInput is embedded (MiniLM) and mapped against **JSON-LD Ontology** to resolve business jargon.")
with c3:
    st.info("**3. SQL Synthesis**\nLlama-3.3 synthesizes BigQuery SQL using the mapped schema guardrails.")
with c4:
    st.info("**4. Cloud Execution**\nServerless execution on **Google BigQuery** with results returned as Pandas DataFrames.")

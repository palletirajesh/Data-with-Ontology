import os
import time
import json
import re
import logging
from datetime import datetime, date
import streamlit as st
import pandas as pd
import requests
from github import Github, Auth
from google.cloud import bigquery
from google.oauth2 import service_account
from sentence_transformers import SentenceTransformer, util

# --- 0. SUPPRESS LOG NOISE ---
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# --- 1. CONFIG & STYLE ---
st.set_page_config(page_title="Risk Data Agent", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .stButton>button { border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
TOGETHER_API_KEY = st.secrets.get("TOGETHER_API_KEY1", "") 
BQ_PROJECT = st.secrets["bigquery"]["project_id"]
BQ_DATASET = st.secrets["bigquery"]["dataset_id"]
HISTORY_TABLE = f"{BQ_PROJECT}.{BQ_DATASET}.query_history"

# --- 2. ENGINES ---
def get_bq_client():
    info = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(info)
    return bigquery.Client(credentials=credentials, project=info["project_id"])

bq_client = get_bq_client()

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

# --- 3. LLM ROUTER (Primary & Fallback) ---
def call_llm_with_fallback(prompt):
    url_groq = "https://api.groq.com/openai/v1/chat/completions"
    headers_groq = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
    
    try:
        res = requests.post(url_groq, headers=headers_groq, json=payload, timeout=12)
        if res.status_code == 200: return res.json()["choices"][0]["message"]["content"]
    except: pass

    if not TOGETHER_API_KEY: return ""
    url_tog = "https://api.together.xyz/v1/chat/completions"
    headers_tog = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
    payload_tog = {"model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
    
    try:
        res_tog = requests.post(url_tog, headers=headers_tog, json=payload_tog, timeout=15)
        return res_tog.json()["choices"][0]["message"]["content"] if res_tog.status_code == 200 else ""
    except: return ""

# --- 4. CORE FUNCTIONS ---

def load_persistent_history():
    try:
        query = f"SELECT user_query, generated_sql FROM `{HISTORY_TABLE}` ORDER BY timestamp DESC LIMIT 25"
        return bq_client.query(query).to_dataframe()
    except: return pd.DataFrame(columns=["user_query", "generated_sql"])

def save_query_to_db(user_text, sql):
    try:
        rows = [{"user_query": user_text, "generated_sql": sql, "timestamp": datetime.now().isoformat()}]
        bq_client.insert_rows_json(HISTORY_TABLE, rows)
    except: pass

def build_context_string():
    context = ""
    try:
        with open("database_schema.md", "r") as f: context += f"--- SCHEMA ---\n{f.read()}\n\n"
        with open("knowledge_base.jsonld", "r") as f: context += f"--- ONTOLOGY ---\n{f.read()}\n"
    except: pass
    return context

def evaluate_and_update_ontology(user_text, original_sql, edited_sql):
    """
    Analyzes SQL corrections to update the JSON-LD Graph.
    Follows: Column -> bank:representsConcept -> bank:Concept -> bank:businessJargon
    """
    # Load current ontology to teach the LLM the current relationships
    try:
        with open("knowledge_base.jsonld", "r") as f:
            ontology_data = f.read()
    except:
        ontology_data = "Ontology file missing."

    prompt = f"""
    You are an Ontology Engineer for a Banking Data Warehouse.
    
    CONTEXT ONTOLOGY (JSON-LD):
    {ontology_data[:3000]} 

    USER INTERACTION:
    Question: "{user_text}"
    AI Original SQL: {original_sql}
    User Corrected SQL: {edited_sql}

    TASK:
    Analyze the user's SQL correction. Your goal is to update the JSON-LD Knowledge Graph.
    
    SCENARIOS:
    1. If the user corrected a Column choice: Find the Concept linked to that column. Add any new terms from the question to that Concept's "bank:businessJargon" array.
    2. If a new logic was introduced: Create a new "bank:Concept" and link the relevant "bank:Column" to it using "bank:representsConcept".
    3. If the correction was just formatting: Return "MISMATCH".

    OUTPUT:
    Return a JSON array of NEW or UPDATED nodes for the "@graph". 
    Use the "bank:" prefix for @ids.
    Example: 
    [
      {{
        "@id": "bank:Concept_Debt",
        "@type": "bank:Concept",
        "bank:businessJargon": ["used amount", "current balance", "debt"]
      }}
    ]

    Return ONLY the raw JSON array. No preamble.
    """
    
    response = call_llm_with_fallback(prompt)
    
    if not response or response.strip().upper() == "MISMATCH":
        return {"status": "warning", "msg": "Ontology Update Skipped: No new business logic detected."}

    try:
        # Clean response and parse JSON
        clean = response.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\[.*\]', clean, re.DOTALL)
        if not match: return {"status": "error", "msg": "AI failed to generate structural JSON."}
        
        new_nodes = json.loads(match.group(0))

        # GitHub Connection
        auth = Auth.Token(st.secrets["github"]["token"])
        g = Github(auth=auth)
        repo = g.get_repo(st.secrets["github"]["repo"])
        
        # Load Existing File from Git
        contents = repo.get_contents("knowledge_base.jsonld", ref="main")
        kb = json.loads(contents.decoded_content.decode("utf-8"))
        graph = kb.get("@graph", [])

        # MERGE LOGIC: Update existing nodes or append new ones
        updated_count = 0
        for new_node in new_nodes:
            new_node["dateAdded"] = date.today().isoformat()
            new_node["reviewStatus"] = "Pending_Review"
            
            # Check if node already exists in graph to merge businessJargon
            found = False
            for existing_node in graph:
                if existing_node.get("@id") == new_node.get("@id"):
                    # Merge jargon sets to avoid duplicates
                    if "bank:businessJargon" in new_node and "bank:businessJargon" in existing_node:
                        combined = list(set(existing_node["bank:businessJargon"]) | set(new_node["bank:businessJargon"]))
                        existing_node["bank:businessJargon"] = combined
                    # Update other fields
                    existing_node.update({k: v for k, v in new_node.items() if k != "bank:businessJargon"})
                    found = True
                    updated_count += 1
                    break
            
            if not found:
                graph.append(new_node)
                updated_count += 1

        kb["@graph"] = graph
        
        # Push to GitHub
        repo.update_file(
            path=contents.path,
            message=f"🧠 Ontology Sync: {user_text[:20]}",
            content=json.dumps(kb, indent=2),
            sha=contents.sha,
            branch="main"
        )
        return {"status": "success", "msg": f"Ontology Updated: {updated_count} nodes synchronized."}
        
    except Exception as e:
        return {"status": "error", "msg": f"Ontology Sync Failed: {e}"}

# --- 5. UI & SIDEBAR ---
if "db_history" not in st.session_state:
    st.session_state.db_history = load_persistent_history()

with st.sidebar:
    st.header("🕒 History")
    if st.button("🔄 Sync with BigQuery", width='stretch'):
        st.session_state.db_history = load_persistent_history()
    
    for idx, row in st.session_state.db_history.iterrows():
        if st.button(row['user_query'], key=f"h_{idx}", width='stretch'):
            st.session_state.main_input = row['user_query']
            st.session_state.sql_editor_key = row['generated_sql']
            st.session_state.last_user_input = row['user_query']
            st.session_state.original_generated_sql = row['generated_sql']
            st.rerun()

# --- 6. MAIN WORKFLOW ---
st.title("🏦 Risk Data Agent")

# Persist feedback result across rerun
if "retrain_result" in st.session_state:
    res = st.session_state.retrain_result
    if res["status"] == "success": st.success(res["msg"])
    elif res["status"] == "warning": st.warning(res["msg"])
    else: st.error(res["msg"])
    del st.session_state.retrain_result

user_input = st.text_input("What risk data do you need?", key="main_input")

if user_input:
    st.session_state["last_user_input_preserved"] = user_input
    col_sql, col_res = st.columns([1, 1.5])
    
    with col_sql:
        if "last_user_input" not in st.session_state or st.session_state.last_user_input != user_input:
            # Semantic Cache
            cached_sql = None
            if not st.session_state.db_history.empty:
                scores = util.cos_sim(embedder.encode(user_input), embedder.encode(st.session_state.db_history['user_query'].tolist()))[0]
                if scores.max() > 0.94: cached_sql = st.session_state.db_history.iloc[scores.argmax().item()]['generated_sql']

            if cached_sql:
                st.toast("⚡ Using Cached Logic")
                final_sql = cached_sql
            else:
                with st.spinner("Synthesizing SQL..."):
                    context = build_context_string()
                    full_path = f"{BQ_PROJECT}.{BQ_DATASET}"
                    # THE 19 RULES
                    system_prompt = (
                        "You are a BigQuery SQL Expert.\n\nContext:\n" + context + "\n\n"
                        "STRICT RULES:\n1. Use Context only. 2. No SELECT *. 3. Always select: cust_id, customer_name, card_id.\n"
                        "4. Join tables as needed. 5. Prefix: " + full_path + ". 6. Use aliases.\n"
                        "7. RAW SQL ONLY. No explanations.\n\nTask: " + user_input
                    )
                    final_sql = call_llm_with_fallback(system_prompt).replace("```sql", "").replace("```", "").strip()
                    save_query_to_db(user_input, final_sql)
            
            st.session_state.last_user_input = user_input
            st.session_state.original_generated_sql = final_sql
            st.session_state.sql_editor_key = final_sql
            st.session_state.pending_feedback = False

        with st.form("sql_form"):
            user_sql = st.text_area("SQL Preview:", value=st.session_state.get('sql_editor_key', ''), height=250)
            if st.form_submit_button("▶️ Execute"):
                try:
                    job = bq_client.query(user_sql)
                    st.session_state.last_df = job.result(timeout=30).to_dataframe(create_bqstorage_client=False)
                    if user_sql.strip() != st.session_state.original_generated_sql.strip():
                        st.session_state.pending_feedback = True
                        st.session_state.edited_sql_for_feedback = user_sql
                except Exception as e: st.error(f"BQ Error: {e}")

    with col_res:
        if st.session_state.get("pending_feedback"):
            st.info("💡 **Retrain model with your SQL edits?**")
            b1, b2, _ = st.columns([1.5, 1.5, 4])
            with b1:
                if st.button("🧠 Yes, Retrain", width='stretch'):
                    _q = st.session_state.get("last_user_input_preserved", "")
                    _o = st.session_state.original_generated_sql
                    _e = st.session_state.edited_sql_for_feedback
                    st.session_state.retrain_result = evaluate_and_update_ontology(_q, _o, _e)
                    st.session_state.pending_feedback = False
                    st.rerun()
            with b2:
                if st.button("❌ No, Ignore", width='stretch'):
                    st.session_state.pending_feedback = False
                    st.rerun()

        if "last_df" in st.session_state:
            st.dataframe(st.session_state.last_df, use_container_width=True)

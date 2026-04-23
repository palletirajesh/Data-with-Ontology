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
    Advanced Logic: Updates Graph nodes based on specific Banking Ontology rules.
    Links bank:Column -> bank:representsConcept -> bank:Concept -> bank:businessJargon
    """
    try:
        with open("knowledge_base.jsonld", "r") as f:
            ontology_context = f.read()
    except:
        ontology_context = "Ontology file missing."

    prompt = f"""
    You are an Ontology Engineer for a Banking Risk Data Warehouse.
    
    CONTEXT ONTOLOGY:
    {ontology_context[:3000]}

    USER INPUT: "{user_text}"
    AI SQL: {original_sql}
    USER CORRECTED SQL: {edited_sql}

    TASK: Analyze why the user corrected the SQL. Update the JSON-LD '@graph'.
    
    STRICT ONTOLOGY RULES:
    1. If the user fixed a filter (e.g., changed column name or value): 
       - Find the Concept that represents that business term.
       - Add the user's specific wording to that Concept's "bank:businessJargon" set.
    2. If the user introduced a new logical Concept:
       - Create a new node with "@type": "bank:Concept".
       - Create a node for the "@type": "bank:Column" used.
       - Link them using "bank:representsConcept".
    3. Use the "bank:" prefix for all new @ids.
    4. If the edit is trivial (whitespace), return: MISMATCH.

    Output ONLY the JSON array of the specific nodes to be added/updated in the graph.
    """
    
    response = call_llm_with_fallback(prompt)
    if not response or "MISMATCH" in response.upper():
        return {"status": "warning", "msg": "Retrain Skipped: No business logic change detected."}

    try:
        clean = response.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\[.*\]', clean, re.DOTALL)
        if not match: return {"status": "error", "msg": "AI failed to generate valid graph nodes."}
        
        new_nodes = json.loads(match.group(0))

        # GitHub Sync
        auth = Auth.Token(st.secrets["github"]["token"])
        g = Github(auth=auth)
        repo = g.get_repo(st.secrets["github"]["repo"])
        contents = repo.get_contents("knowledge_base.jsonld", ref="main")
        kb = json.loads(contents.decoded_content.decode("utf-8"))
        graph = kb.get("@graph", [])

        # Merge new nodes into existing graph
        for new_node in new_nodes:
            new_node["dateAdded"] = date.today().isoformat()
            new_node["reviewStatus"] = "Pending_Review"
            
            # Check for existing @id to merge jargon instead of duplicating
            found = False
            for existing in graph:
                if existing.get("@id") == new_node.get("@id"):
                    if "bank:businessJargon" in new_node and "bank:businessJargon" in existing:
                        existing["bank:businessJargon"] = list(set(existing["bank:businessJargon"]) | set(new_node["bank:businessJargon"]))
                    existing.update({k: v for k, v in new_node.items() if k != "bank:businessJargon"})
                    found = True; break
            if not found: graph.append(new_node)

        kb["@graph"] = graph
        repo.update_file(contents.path, f"🧠 Ontology Retrain: {user_text[:20]}", json.dumps(kb, indent=2), contents.sha, branch="main")
        return {"status": "success", "msg": "GitHub Knowledge Base Updated Successfully!"}
    except Exception as e:
        return {"status": "error", "msg": f"GitHub Error: {e}"}

# --- 5. SIDEBAR ---
if "db_history" not in st.session_state:
    st.session_state.db_history = load_persistent_history()

with st.sidebar:
    st.header("🕒 Query History")
    if st.button("🔄 Sync with BigQuery", width='stretch'):
        st.session_state.db_history = load_persistent_history()
    
    st.markdown("---")
    for idx, row in st.session_state.db_history.iterrows():
        # Clicking restores full text and SQL to UI
        if st.button(row['user_query'], key=f"h_{idx}", width='stretch'):
            st.session_state.main_input = row['user_query']
            st.session_state.sql_editor_key = row['generated_sql']
            st.session_state.last_user_input = row['user_query']
            st.session_state.original_generated_sql = row['generated_sql']
            st.rerun()

# --- 6. MAIN WORKFLOW ---
st.title("🏦 Risk Data Agent")

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
        st.subheader("⚙️ Generated Logic")
        
        if "last_user_input" not in st.session_state or st.session_state.last_user_input != user_input:
            # Semantic Cache
            cached_sql = None
            if not st.session_state.db_history.empty:
                scores = util.cos_sim(embedder.encode(user_input), embedder.encode(st.session_state.db_history['user_query'].tolist()))[0]
                if scores.max() > 0.94: cached_sql = st.session_state.db_history.iloc[scores.argmax().item()]['generated_sql']

            if cached_sql:
                st.toast("⚡ Reusing Logic from Cache")
                final_sql = cached_sql
            else:
                with st.spinner("Synthesizing SQL (19-Rule Mode)..."):
                    context = build_context_string()
                    full_path = f"{BQ_PROJECT}.{BQ_DATASET}"
                    
                    # --- RESTORED FULL 19 RULES PROMPT ---
                    system_prompt = (
                        "You are a BigQuery SQL Expert.\n\nContext:\n" + context + "\n\n"
                        "STRICT RULES:\n"
                        "1. ONLY use Tables/Columns in Context.\n"
                        "2. NEVER USE 'SELECT *'. Explicitly name columns.\n"
                        "3. DEFAULT COLUMNS: Always select: `cust_id`, `customer_name`, `card_id`.\n"
                        "4. DYNAMIC COLUMNS: Select columns relating to user conditions.\n"
                        "5. PROACTIVE JOINS: Write JOINs for required tables.\n"
                        "7. NO PARENTHESES: Never put () after table names.\n"
                        "8. Use MANDATORY JOINs exactly.\n"
                        "9. Output ONLY raw SQL code.\n"
                        "10. If missing, output: 'I cannot answer this with available data.'\n"
                        "11. dim_customer is the bridge table.\n"
                        f"12. Prefix tables with: `{full_path}.`\n"
                        "15. For filters, use: UPPER(column) = UPPER('value').\n"
                        "16. Apply BUSINESS TRANSLATION RULES strictly.\n"
                        "17. Any column in WHERE/HAVING must be in SELECT.\n"
                        "18. Return detailed records unless summary keywords are used.\n"
                        "19. Assign short aliases (t1, t2) and prefix EVERY column.\n\n"
                        f"Write BigQuery SQL for: \"{user_input}\""
                    )
                    final_sql = call_llm_with_fallback(system_prompt).replace("```sql", "").replace("```", "").strip()
                    save_query_to_db(user_input, final_sql)
            
            st.session_state.last_user_input = user_input
            st.session_state.original_generated_sql = final_sql
            st.session_state.sql_editor_key = final_sql
            st.session_state.pending_feedback = False

        with st.form("sql_form"):
            user_sql = st.text_area("SQL Preview:", value=st.session_state.get('sql_editor_key', ''), height=250)
            if st.form_submit_button("▶️ Execute Query"):
                try:
                    job = bq_client.query(user_sql)
                    st.session_state.last_df = job.result(timeout=30).to_dataframe(create_bqstorage_client=False)
                    if user_sql.strip() != st.session_state.original_generated_sql.strip():
                        st.session_state.pending_feedback = True
                        st.session_state.edited_sql_for_feedback = user_sql
                except Exception as e: st.error(f"BQ Error: {e}")

    with col_res:
        st.subheader("📊 Results")
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
                if st.button("❌ No", width='stretch'):
                    st.session_state.pending_feedback = False
                    st.rerun()

        if "last_df" in st.session_state:
            st.dataframe(st.session_state.last_df, use_container_width=True)

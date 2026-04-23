import os
import time
import json
import re
from datetime import datetime, date
import streamlit as st
import pandas as pd
import requests
from github import Github, Auth  # Modern GitHub Auth for v10+
from google.cloud import bigquery
from google.oauth2 import service_account
from sentence_transformers import SentenceTransformer, util

# --- 1. CONFIG & STYLE ---
st.set_page_config(page_title="Risk Data Agent", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .stDownloadButton>button { width: 100%; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# Secrets Management
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
TOGETHER_API_KEY = st.secrets.get("TOGETHER_API_KEY1", "") # Using the correct key from your secrets
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

# --- 3. FAILOVER & LLM ROUTER ---

def call_llm_with_fallback(prompt):
    """Primary: Groq | Fallback: Together AI (Llama-3.3-70B)"""
    url_groq = "https://api.groq.com/openai/v1/chat/completions"
    headers_groq = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload_groq = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    
    try:
        res = requests.post(url_groq, headers=headers_groq, json=payload_groq, timeout=12)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        elif res.status_code == 429:
            st.toast("⚠️ Groq Rate Limit. Switching to Fallback Provider...")
    except Exception:
        st.toast("📡 Groq Connection Failed. Attempting Fallback...")

    # --- FALLBACK TO TOGETHER AI ---
    if not TOGETHER_API_KEY:
        st.error("Fallback Failed: TOGETHER_API_KEY1 missing in Secrets.")
        return ""

    url_tog = "https://api.together.xyz/v1/chat/completions"
    headers_tog = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
    payload_tog = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    
    try:
        res_tog = requests.post(url_tog, headers=headers_tog, json=payload_tog, timeout=15)
        if res_tog.status_code == 200:
            st.toast("✅ Fallback Successful (Together AI)")
            return res_tog.json()["choices"][0]["message"]["content"]
        return ""
    except Exception:
        return ""

# --- 4. PERSISTENCE & GITHUB UPDATES ---

def load_persistent_history():
    """Fetch query history from BigQuery."""
    try:
        query = f"SELECT user_query, generated_sql FROM `{HISTORY_TABLE}` ORDER BY timestamp DESC LIMIT 25"
        return bq_client.query(query).to_dataframe()
    except Exception:
        return pd.DataFrame(columns=["user_query", "generated_sql"])

def save_query_to_db(user_text, sql):
    """Saves to BigQuery with explicit error reporting."""
    try:
        rows_to_insert = [{
            "user_query": user_text,
            "generated_sql": sql,
            "timestamp": datetime.now().isoformat()
        }]
        errors = bq_client.insert_rows_json(HISTORY_TABLE, rows_to_insert)
        if errors:
            st.error(f"BQ Save Error (Column mismatch?): {errors}")
        else:
            st.toast("✅ Saved to History")
    except Exception as e:
        st.error(f"Save Failed: {e}")

def build_context_string():
    """Reads FULL Schema and Ontology files for maximum AI accuracy."""
    context = ""
    # RESTORED FILENAME: database_schema.md
    try:
        with open("database_schema.md", "r") as f:
            context += f"--- DATABASE SCHEMA ---\n{f.read()}\n\n"
    except: pass
    
    # FILENAME: knowledge_base.jsonld
    try:
        with open("knowledge_base.jsonld", "r") as f:
            context += f"--- ONTOLOGY ---\n{f.read()}\n"
    except: pass
    return context

def evaluate_and_update_ontology(user_text, original_sql, edited_sql):
    """Pushes logic corrections back to GitHub (knowledge_base.jsonld)."""
    with st.spinner("Analyzing edits for retraining..."):
        prompt = f"User asked '{user_text}'. AI wrote '{original_sql}'. User corrected to '{edited_sql}'. Extract business jargon mapping as a raw JSON array. If nothing new, return 'MISMATCH'."
        response = call_llm_with_fallback(prompt)
    
    if "MISMATCH" in response:
        st.warning("Retrain Skipped: AI found no new business logic in your edits.")
        return

    try:
        # Robust JSON cleaning
        clean = response.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\[.*\]', clean, re.DOTALL)
        if not match: 
            st.error("AI returned invalid mapping format. Check logs.")
            return
        
        new_mappings = json.loads(match.group(0))
        today = date.today().isoformat()
        for m in new_mappings:
            m["dateAdded"] = today
            m["reviewStatus"] = "Pending_Review"
        
        # Modern GitHub Auth (Fixes DeprecationWarning in logs)
        auth = Auth.Token(st.secrets["github"]["token"])
        g = Github(auth=auth)
        repo_name = st.secrets["github"]["repo"]
        repo = g.get_repo(repo_name)
        
        file_path = "knowledge_base.jsonld"
        contents = repo.get_contents(file_path, ref="main")
        kb = json.loads(contents.decoded_content.decode("utf-8"))
        kb.setdefault("@graph", []).extend(new_mappings)
        
        repo.update_file(
            path=contents.path,
            message=f"🤖 AI Retrain: {user_text[:20]}",
            content=json.dumps(kb, indent=2),
            sha=contents.sha,
            branch="main"
        )
        st.success(f"✅ GitHub Updated: {len(new_mappings)} new rules added!")
        time.sleep(1)
    except Exception as e:
        st.error(f"❌ GitHub Sync Failed: {e}")

# --- 5. UI & SIDEBAR SYNC ---
if "db_history" not in st.session_state:
    st.session_state.db_history = load_persistent_history()

with st.sidebar:
    st.header("🕒 Query History")
    if st.button("🔄 Sync with BigQuery"):
        st.session_state.db_history = load_persistent_history()
    
    st.markdown("---")
    for idx, row in st.session_state.db_history.iterrows():
        # ON CLICK: Restore the full UI state
        if st.button(row['user_query'], key=f"h_{idx}", use_container_width=True):
            st.session_state.main_input = row['user_query']
            st.session_state.sql_editor_key = row['generated_sql']
            st.session_state.last_user_input = row['user_query']
            st.session_state.original_generated_sql = row['generated_sql']
            st.rerun()

# --- 6. MAIN WORKFLOW ---
st.title("🏦 Risk Data Agent")
# Linked to 'main_input' for sidebar sync
user_input = st.text_input("What risk data do you need?", key="main_input", placeholder="e.g. Amazon customers with late payments...")

if user_input:
    col_sql, col_res = st.columns([1, 1.5])
    
    with col_sql:
        st.subheader("⚙️ Generated Logic")
        
        # Check if we need to call the AI OR load from memory
        if "last_user_input" not in st.session_state or st.session_state.last_user_input != user_input:
            
            # Semantic Cache Check
            cached_sql = None
            if not st.session_state.db_history.empty:
                curr_emb = embedder.encode(user_input, convert_to_tensor=True)
                past_embs = embedder.encode(st.session_state.db_history['user_query'].tolist(), convert_to_tensor=True)
                scores = util.cos_sim(curr_emb, past_embs)[0]
                if scores.max() > 0.94:
                    cached_sql = st.session_state.db_history.iloc[scores.argmax().item()]['generated_sql']

            if cached_sql:
                st.toast("⚡ Semantic Match Found! Reusing Logic.")
                final_sql = cached_sql
            else:
                with st.spinner("Synthesizing Logic..."):
                    context = build_context_string()
                    full_path = f"{BQ_PROJECT}.{BQ_DATASET}"
                    # FULL 19 RULES RESTORED
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

        # SQL Editor Form
        with st.form("editor_form"):
            user_sql = st.text_area("SQL Preview:", value=st.session_state.get('sql_editor_key', ''), height=250)
            if st.form_submit_button("▶️ Execute Query"):
                try:
                    job = bq_client.query(user_sql)
                    st.session_state.last_df = job.result(timeout=30).to_dataframe(create_bqstorage_client=False)
                    # Trigger retraining prompt if user edited the SQL
                    if user_sql.strip() != st.session_state.original_generated_sql.strip():
                        st.session_state.pending_feedback = True
                        st.session_state.edited_sql_for_feedback = user_sql
                except Exception as e:
                    st.error(f"BQ Error: {e}")

    with col_res:
        st.subheader("📊 Results")
        if st.session_state.get("pending_feedback"):
            st.info("💡 **Modified query detected. Retrain the model?**")
            b1, b2, _ = st.columns([1.5, 1.5, 4])
            with b1:
                if st.button("🧠 Yes, update ontology", use_container_width=True):
                    evaluate_and_update_ontology(user_input, st.session_state.original_generated_sql, st.session_state.edited_sql_for_feedback)
                    st.session_state.pending_feedback = False
                    st.rerun()
            with b2:
                if st.button("❌ No", use_container_width=True):
                    st.session_state.pending_feedback = False
                    st.rerun()

        if "last_df" in st.session_state:
            st.dataframe(st.session_state.last_df, use_container_width=True)

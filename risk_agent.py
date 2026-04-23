import os
import time
import json
import re
from datetime import datetime, date
import streamlit as st
import pandas as pd
import requests
from github import Github
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
TOGETHER_API_KEY = st.secrets.get("TOGETHER_API_KEY", "") # Failover Key
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
    
    # --- TRY GROQ ---
    url_groq = "https://api.groq.com/openai/v1/chat/completions"
    headers_groq = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload_groq = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    
    try:
        res = requests.post(url_groq, headers=headers_groq, json=payload_groq, timeout=15)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        elif res.status_code == 429:
            st.toast("⚠️ Groq Rate Limit. Switching to Fallback Provider...")
        else:
            st.warning(f"Groq issue ({res.status_code}). Attempting Fallback...")
    except Exception:
        st.toast("📡 Groq Connection Failed. Attempting Fallback...")

    # --- TRY TOGETHER AI (The Demo Safety Net) ---
    if not TOGETHER_API_KEY:
        st.error("Fallback Failed: No TOGETHER_API_KEY found in Streamlit Secrets.")
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
        else:
            st.error(f"Critical: Both Providers Failed. {res_tog.text}")
            return ""
    except Exception as e:
        st.error(f"Critical System Failure: {e}")
        return ""

# --- 4. PERSISTENCE & CONTEXT ---

def load_persistent_history():
    """Load history from BigQuery to populate Sidebar and Semantic Cache."""
    try:
        query = f"SELECT user_query, generated_sql FROM `{HISTORY_TABLE}` ORDER BY timestamp DESC LIMIT 25"
        return bq_client.query(query).to_dataframe()
    except Exception:
        return pd.DataFrame(columns=["user_query", "generated_sql"])

def save_query_to_db(user_text, sql):
    """Save generation to BigQuery history."""
    try:
        rows = [{"user_query": user_text, "generated_sql": sql, "timestamp": datetime.now().isoformat()}]
        bq_client.insert_rows_json(HISTORY_TABLE, rows)
    except Exception as e:
        st.warning(f"Note: History log failed: {e}")

def build_context_string():
    """Reads FULL Schema and Ontology files for maximum accuracy."""
    context = ""
    try:
        with open("database_schema.md", "r") as f:
            context += f"--- DATABASE SCHEMA ---\n{f.read()}\n\n"
    except: pass
    try:
        with open("knowledge_base.jsonld", "r") as f:
            context += f"--- ONTOLOGY & JARGON MAPPING ---\n{f.read()}\n"
    except: pass
    return context

def evaluate_and_update_ontology(user_text, original_sql, edited_sql):
    """Pushes logic corrections back to GitHub."""
    prompt = f"User asked '{user_text}'. AI wrote '{original_sql}'. User corrected to '{edited_sql}'. Extract business jargon mapping as a raw JSON array. If nothing new, return 'MISMATCH'."
    response = call_llm_with_fallback(prompt)
    
    if "MISMATCH" not in response:
        try:
            clean = response.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\[.*\]', clean, re.DOTALL)
            if not match: return
            
            new_mappings = json.loads(match.group(0))
            today = date.today().isoformat()
            for m in new_mappings:
                m["dateAdded"] = today
                m["reviewStatus"] = "Pending_Review"
            
            # GitHub Push
            repo = Github(st.secrets["github"]["token"]).get_repo("palletirajesh/Data-with-Ontology")
            contents = repo.get_contents("knowledge_base.jsonld", ref="main")
            kb = json.loads(contents.decoded_content.decode("utf-8"))
            kb.setdefault("@graph", []).extend(new_mappings)
            
            repo.update_file(contents.path, f"🤖 AI Retrain: {user_text[:20]}", json.dumps(kb, indent=2), contents.sha, branch="main")
            st.success("✅ Knowledge pushed to GitHub!")
            time.sleep(2)
        except Exception as e:
            st.error(f"GitHub Update Failed: {e}")

# --- 5. UI & SIDEBAR ---

if "db_history" not in st.session_state:
    st.session_state.db_history = load_persistent_history()

with st.sidebar:
    st.header("🕒 Query History")
    if st.button("🔄 Refresh History"):
        st.session_state.db_history = load_persistent_history()
    
    for idx, row in st.session_state.db_history.iterrows():
        if st.button(f"🔍 {row['user_query'][:35]}...", key=f"hist_{idx}", use_container_width=True):
            st.session_state.last_user_input = row['user_query']
            st.session_state.sql_editor_key = row['generated_sql']
            st.session_state.original_generated_sql = row['generated_sql']
            st.rerun()

# --- 6. MAIN WORKFLOW ---

st.title("🏦 Risk Data Agent")
user_input = st.text_input("What data do you need?", placeholder="e.g. Amazon customers with late payments...", key="main_input")

if user_input:
    col_sql, col_res = st.columns([1, 1.5])
    
    with col_sql:
        st.subheader("⚙️ Generated Logic")
        
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
                st.toast("⚡ Semantic Match Found! (BQ Cache)")
                final_sql = cached_sql
            else:
                with st.spinner("Synthesizing Logic (Failover Active)..."):
                    context = build_context_string()
                    full_path = f"{BQ_PROJECT}.{BQ_DATASET}"
                    # THE 19 RULES PROMPT
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
            
            # Sync session state
            st.session_state.last_user_input = user_input
            st.session_state.original_generated_sql = final_sql
            st.session_state.sql_editor_key = final_sql
            st.session_state.pending_feedback = False

        # SQL Editor
        with st.form("editor_form"):
            user_sql = st.text_area("SQL Editor:", value=st.session_state.get('sql_editor_key', ''), height=250)
            if st.form_submit_button("▶️ Execute Query"):
                try:
                    job = bq_client.query(user_sql)
                    st.session_state.last_df = job.result(timeout=30).to_dataframe(create_bqstorage_client=False)
                    if user_sql.strip() != st.session_state.original_generated_sql.strip():
                        st.session_state.pending_feedback = True
                        st.session_state.edited_sql_for_feedback = user_sql
                except Exception as e:
                    st.error(f"BQ Error: {e}")

    with col_res:
        st.subheader("📊 Results")
        
        if st.session_state.get("pending_feedback"):
            st.info("💡 **Retrain model with your SQL edits?**")
            b1, b2, _ = st.columns([1.5, 1.5, 4])
            with b1:
                if st.button("🧠 Yes", use_container_width=True):
                    evaluate_and_update_ontology(user_input, st.session_state.original_generated_sql, st.session_state.edited_sql_for_feedback)
                    st.session_state.pending_feedback = False
                    st.rerun()
            with b2:
                if st.button("❌ No", use_container_width=True):
                    st.session_state.pending_feedback = False
                    st.rerun()

        if "last_df" in st.session_state:
            st.dataframe(st.session_state.last_df, use_container_width=True)

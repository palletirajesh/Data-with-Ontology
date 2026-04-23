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

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
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

# --- 3. PERSISTENCE & HELPERS ---

def load_persistent_history():
    """Retrieves query history from BigQuery to populate the sidebar and semantic cache."""
    try:
        query = f"SELECT user_query, generated_sql FROM `{HISTORY_TABLE}` ORDER BY timestamp DESC LIMIT 20"
        return bq_client.query(query).to_dataframe()
    except Exception as e:
        return pd.DataFrame(columns=["user_query", "generated_sql"])

def save_query_to_db(user_text, sql):
    """Saves a successfully generated query to the BigQuery history table."""
    try:
        rows_to_insert = [{
            "user_query": user_text,
            "generated_sql": sql,
            "timestamp": datetime.now().isoformat()
        }]
        bq_client.insert_rows_json(HISTORY_TABLE, rows_to_insert)
    except Exception as e:
        st.warning(f"Note: Could not log to BigQuery history: {e}")

def call_groq_llm(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        st.error(f"LLM Error: {response.text}")
        return ""

def build_context_string():
    """Loads the FULL schema and ontology files without trimming."""
    context = ""
    try:
        with open("database_schema_23ap.md", "r") as f:
            context += f"--- DATABASE SCHEMA ---\n{f.read()}\n\n"
    except: pass
    try:
        with open("knowledge_base_23Ap.jsonld", "r") as f:
            context += f"--- ONTOLOGY & JARGON MAPPING ---\n{f.read()}\n"
    except: pass
    return context

def evaluate_and_update_ontology(user_text, original_sql, edited_sql):
    """Pushes new learned mappings to GitHub main branch."""
    evaluation_prompt = f"""
    The user asked: "{user_text}"
    The AI generated: "{original_sql}"
    The User corrected to: "{edited_sql}"
    Did the user fix a table join, column mapping, or business jargon? 
    If YES, extract the new mapping as a valid JSON array of objects using our bank ontology.
    Return ONLY the raw JSON array string. NO markdown backticks or text.
    If NO, return exactly "MISMATCH".
    """
    response = call_groq_llm(evaluation_prompt) 
    if "MISMATCH" not in response:
        try:
            clean_response = response.replace("```json", "").replace("```", "").strip()
            json_match = re.search(r'\[.*\]', clean_response, re.DOTALL)
            if not json_match: return

            new_data = json.loads(json_match.group(0))
            if isinstance(new_data, dict): new_data = [new_data]
            
            today_str = date.today().isoformat()
            for item in new_data:
                item["dateAdded"] = today_str
                item["reviewStatus"] = "Pending_Review"
            
            repo = Github(st.secrets["github"]["token"]).get_repo("palletirajesh/Data-with-Ontology")
            contents = repo.get_contents("knowledge_base_23Ap.jsonld", ref="main")
            kb_data = json.loads(contents.decoded_content.decode("utf-8"))
            
            kb_data.setdefault("@graph", []).extend(new_data)
            
            repo.update_file(
                contents.path,
                f"🤖 AI Retrain: {user_text[:30]}...",
                json.dumps(kb_data, indent=2),
                contents.sha,
                branch="main"
            )
            st.success("✅ Knowledge successfully pushed to GitHub!")
            time.sleep(2)
        except Exception as e:
            st.error(f"GitHub Sync Failed: {e}")

# --- 4. SIDEBAR & STATE ---
if "db_history" not in st.session_state:
    st.session_state.db_history = load_persistent_history()

with st.sidebar:
    st.header("🕒 Query History")
    st.caption("Last 20 Persistent Queries")
    if st.button("🔄 Refresh History"):
        st.session_state.db_history = load_persistent_history()
    
    for idx, row in st.session_state.db_history.iterrows():
        if st.button(f"🔍 {row['user_query'][:35]}...", key=f"hist_{idx}", use_container_width=True):
            st.session_state.last_user_input = row['user_query']
            st.session_state.sql_editor_key = row['generated_sql']
            st.session_state.original_generated_sql = row['generated_sql']
            st.rerun()

# --- 5. MAIN APP UI ---
st.title("🏦 Risk Data Agent")
user_input = st.text_input("What risk data do you need?", key="main_input", placeholder="e.g., Amazon customers with late payments...")

if user_input:
    col_sql, col_res = st.columns([1, 1.5])
    
    with col_sql:
        st.subheader("⚙️ Generated Logic")
        
        # Determine if we need to generate new SQL or use Cache
        if "last_user_input" not in st.session_state or st.session_state.last_user_input != user_input:
            
            # Semantic Cache Check
            cached_sql = None
            if not st.session_state.db_history.empty:
                embeddings_past = embedder.encode(st.session_state.db_history['user_query'].tolist(), convert_to_tensor=True)
                embedding_curr = embedder.encode(user_input, convert_to_tensor=True)
                scores = util.cos_sim(embedding_curr, embeddings_past)[0]
                best_idx = scores.argmax().item()
                if scores[best_idx] > 0.94:
                    cached_sql = st.session_state.db_history.iloc[best_idx]['generated_sql']

            if cached_sql:
                st.toast("⚡ Semantic Match Found! Bypassing Groq.")
                final_sql = cached_sql
            else:
                with st.spinner("Translating intent via Llama-3..."):
                    context = build_context_string()
                    full_path = f"{BQ_PROJECT}.{BQ_DATASET}"
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
                        "15. For string type filters, use: UPPER(column) = UPPER('value').\n"
                        "16. Apply BUSINESS TRANSLATION RULES strictly.\n"
                        "17. Any column in WHERE/HAVING must be in SELECT.\n"
                        "18. Return detailed records unless summary keywords (count, total) are used.\n"
                        "19. Assign short aliases (t1, t2) and prefix EVERY column.\n\n"
                        f"Write BigQuery SQL for: \"{user_input}\""
                    )
                    final_sql = call_groq_llm(system_prompt).replace("```sql", "").replace("```", "").strip()
                    save_query_to_db(user_input, final_sql)
            
            st.session_state.last_user_input = user_input
            st.session_state.original_generated_sql = final_sql
            st.session_state.sql_editor_key = final_sql
            st.session_state.pending_feedback = False

        # SQL Editor Form
        with st.form("sql_editor_form"):
            user_edited_sql = st.text_area("Review/Edit SQL:", value=st.session_state.get('sql_editor_key', ''), height=250)
            execute_btn = st.form_submit_button("▶️ Execute Query")
            
        if execute_btn:
            with st.status("🚀 Running BigQuery Job...", expanded=True) as status:
                try:
                    job = bq_client.query(user_edited_sql)
                    df = job.result(timeout=30).to_dataframe(create_bqstorage_client=False)
                    st.session_state.last_df = df
                    status.update(label="✅ Query Finished", state="complete", expanded=False)
                    if user_edited_sql.strip() != st.session_state.original_generated_sql.strip():
                        st.session_state.pending_feedback = True
                except Exception as e:
                    status.update(label="❌ Query Error", state="error")
                    st.error(e)

    with col_res:
        st.subheader("📊 Data Results")
        if st.session_state.get("pending_feedback", False):
            st.info("💡 **Modified query detected. Retrain the model?**")
            b1, b2, _ = st.columns([1.5, 1.5, 4])
            with b1:
                if st.button("🧠 Yes", use_container_width=True):
                    evaluate_and_update_ontology(user_input, st.session_state.original_generated_sql, user_edited_sql)
                    st.session_state.pending_feedback = False
                    st.rerun()
            with b2:
                if st.button("❌ No", use_container_width=True):
                    st.session_state.pending_feedback = False
                    st.rerun()

        if "last_df" in st.session_state:
            st.dataframe(st.session_state.last_df, use_container_width=True)

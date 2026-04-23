import os
import time
import json
from datetime import date
import streamlit as st
import pandas as pd
import re
import requests
import re
from github import Github # Make sure to import this at the top of your file!
import rdflib
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

# --- STATE INITIALIZATION ---
if "query_history" not in st.session_state:
    st.session_state.query_history = [] # For the sidebar
if "semantic_cache" not in st.session_state:
    st.session_state.semantic_cache = [] # For cost reduction
if "selected_past_query" not in st.session_state:
    st.session_state.selected_past_query = ""

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

# --- 3. HELPER FUNCTIONS ---
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

import re # Add this import at the top of your file if not already there

def evaluate_and_update_ontology(user_text, original_sql, edited_sql):
    evaluation_prompt = f"""
    The user asked: "{user_text}"
    The AI generated: "{original_sql}"
    The User corrected to: "{edited_sql}"
    
    Did the user fix a table join, column mapping, or business jargon? 
    If YES, extract the new mapping as a valid JSON array of objects (using our bank ontology).
    ONLY return the JSON array. Do not include explanations or markdown backticks.
    If NO, return exactly "MISMATCH".
    """
    response = call_groq_llm(evaluation_prompt) 
    
    if "MISMATCH" not in response:
        try:
            # 1. Safely extract JSON
            json_match = re.search(r'\[.*\]|\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON found in LLM response.")
                
            new_knowledge_json = json.loads(json_match.group(0))
            if isinstance(new_knowledge_json, dict):
                new_knowledge_json = [new_knowledge_json]
                
            today_str = date.today().isoformat()
            for item in new_knowledge_json:
                item["dateAdded"] = today_str
                item["reviewStatus"] = "Pending_Review"
            
            # --- THE GITHUB FIX ---
            # Make sure this matches the EXACT name of the file in your GitHub repo
            file_path = "knowledge_base_23Ap_updated_v1.jsonld" 
            
            # Connect to GitHub
            g = Github(st.secrets["github"]["token"])
            repo = g.get_repo(st.secrets["github"]["repo"])
            
            # Fetch the current file from GitHub
            contents = repo.get_contents(file_path)
            kb_data = json.loads(contents.decoded_content.decode("utf-8"))
            
            # Append the new AI mappings to the @graph array
            if "@graph" in kb_data:
                kb_data["@graph"].extend(new_knowledge_json)
            else:
                kb_data["@graph"] = new_knowledge_json
                
            updated_json_str = json.dumps(kb_data, indent=2)
            
            # Push the updated file BACK to GitHub
            repo.update_file(
                contents.path,
                f"🤖 Agent Auto-Retrain: Updated Ontology on {today_str}",
                updated_json_str,
                contents.sha
            )
            # ------------------------
            
            st.success(f"✅ Model retrained! Changes pushed directly to GitHub for review.")
            time.sleep(10)
            
        except json.JSONDecodeError:
            st.warning("Feedback loop skipped: AI did not return strictly valid JSON.")
            time.sleep(10)
        except Exception as e:
            st.error(f"Error pushing to GitHub: {e}")
            time.sleep(10)

def build_context_string():
    context = ""
    try:
        with open("database_schema.md", "r") as f:
            context += f"--- DATABASE SCHEMA ---\n{f.read()}\n\n"
    except Exception as e:
        pass
    try:
        with open("knowledge_base.jsonld", "r") as f:
            context += f"--- ONTOLOGY & JARGON MAPPING ---\n{f.read()}\n"
    except Exception as e:
        pass
    return context

# --- 4. SIDEBAR UI (HISTORY) ---
with st.sidebar:
    st.header("🕒 Query History")
    st.markdown("Click a past query to load it.")
    
    if not st.session_state.query_history:
        st.info("No queries run yet.")
    else:
        # Display buttons for the last 10 queries, reversed so newest is on top
        for idx, past_q in enumerate(reversed(st.session_state.query_history[-10:])):
            if st.button(f"🔍 {past_q['user_text']}", key=f"hist_btn_{idx}"):
                st.session_state.selected_past_query = past_q["user_text"]
                st.rerun()

# --- 5. MAIN APP UI ---
st.title("🏦 Risk Data Agent")
st.markdown("Ask natural language questions to generate BigQuery SQL logic.")

# The text input uses the selected history item if clicked, otherwise empty
user_input = st.text_input("What risk data do you need?", value=st.session_state.selected_past_query, placeholder="e.g., Show me Amazon customers with late payments...")

if user_input:
    col_sql, col_res = st.columns([1, 1.5])
    
    with col_sql:
        st.subheader("⚙️ Generated Logic")
        
        is_new_question = ("last_user_input" not in st.session_state or st.session_state.last_user_input != user_input)
        
        if is_new_question:
            with st.spinner("Analyzing intent..."):
                # --- COST OPTIMIZATION: Semantic Cache Check ---
                user_emb = embedder.encode(user_input, convert_to_tensor=True)
                cached_sql = None
                best_score = 0
                
                # Check if we have asked something highly similar before
                for item in st.session_state.semantic_cache:
                    score = util.cos_sim(user_emb, item["embedding"]).item()
                    if score > best_score:
                        best_score = score
                        if score > 0.94: # 94% similarity threshold
                            cached_sql = item["sql"]
                
                if cached_sql:
                    st.toast("⚡ Loaded from Semantic Cache (Cost: $0)")
                    final_sql = cached_sql
                else:
                    # --- If no cache match, run the LLM ---
                    st.toast("🧠 Generating new SQL via Llama-3...")
                    knowledge_context = build_context_string()
                    full_path = f"{BQ_PROJECT}.{BQ_DATASET}"
                    
                    system_prompt = (
                        "You are a BigQuery SQL Expert.\n\n"
                        "Context:\n" + knowledge_context + "\n\n"
                        "STRICT RULES:\n"
                        "1. ONLY use Tables/Columns in Context.\n"
                        "2. NEVER USE 'SELECT *'. You must explicitly name the columns in your SELECT statement.\n"
                        "3. DEFAULT COLUMNS: Whenever a user asks about 'customers' or 'clients', you MUST ALWAYS select at least:\n"
                        "   - `cust_id`\n"
                        "   - `customer_name`\n"
                        "   - `card_id`\n"
                        "4. DYNAMIC COLUMNS: In addition to the default columns, you MUST select the columns that relate to the user's conditions.\n"
                        "5. PROACTIVE JOINS: If picking dynamic columns requires other tables, you MUST write the JOIN.\n"
                        "7. NO PARENTHESES: Never put () after a table name.\n"
                        "8. Use MANDATORY JOINs exactly. Sequence tables logically.\n"
                        "9. Output ONLY raw SQL code. No markdown or explanations.\n"
                        "10. If data is missing, output EXACTLY: \"I cannot answer this with the available data.\"\n"
                        "11. Always begin with a SELECT clause. For multi-table joins, dim_customer should act a bridge table.\n"
                        f"12. EVERY table name in the SQL must be prefixed with: `{full_path}.`\n"
                        f"13. Example format: SELECT t1.cust_id FROM `{full_path}.dim_customer` t1 JOIN `{full_path}.dim_card_association` t2\n"
                        "14. Tables available: dim_customer, dim_card_association, fact_card_ledger, fact_credit_bureau.\n"
                        "15. For string filters, use: UPPER(column) = UPPER('value').\n"
                        "16. TRANSLATION: Apply BUSINESS TRANSLATION RULES strictly to map user jargon to correct columns.\n"
                        "17. TRANSPARENCY RULE (CRITICAL): Any column you use in the WHERE or HAVING clause MUST also be included in the SELECT clause.\n"
                        "18. GRANULARITY: ALWAYS return a detailed list of records (SELECT explicit columns) rather than a summary or count.\n"
                        "19. ALIASES: Whenever writing a query that contains a JOIN, you must assign short table aliases (e.g., t1, t2) and explicitly prefix every single column.\n\n"
                        f"Write BigQuery SQL for this user request: \"{user_input}\""
                    )
                    final_sql = call_groq_llm(system_prompt).replace("```sql", "").replace("```", "").strip()
                    
                    # Save to Semantic Cache for future cost savings
                    st.session_state.semantic_cache.append({
                        "embedding": user_emb,
                        "user_text": user_input,
                        "sql": final_sql
                    })
                
                # Save to UI Sidebar History
                if user_input not in [q["user_text"] for q in st.session_state.query_history]:
                    st.session_state.query_history.append({"user_text": user_input, "sql": final_sql})

                # Update operational state
                st.session_state.last_user_input = user_input
                st.session_state.original_generated_sql = final_sql
                st.session_state.sql_editor_key = final_sql 
                st.session_state.pending_feedback = False 
                if "last_df" in st.session_state:
                    del st.session_state["last_df"]

        st.subheader("📝 Review and Edit SQL")
        
        with st.form("sql_editor_form"):
            user_edited_sql = st.text_area(
                "Modify the query below if needed before running:", 
                key="sql_editor_key", 
                height=250
            )
            execute_btn = st.form_submit_button("▶️ Execute Query")
            
        if execute_btn:
            start_time = time.perf_counter()
            with st.status("🚀 Running BigQuery Job...", expanded=True) as status:
                try:
                    # Fire to BigQuery (If it's the exact same SQL, BQ cache makes it $0 and instant)
                    query_job = bq_client.query(user_edited_sql)
                    query_result = query_job.result(timeout=30)
                    df = query_result.to_dataframe(create_bqstorage_client=False)
                    
                    end_time = time.perf_counter()
                    st.session_state.last_df = df
                    status.update(label=f"✅ Query Finished in {end_time - start_time:.2f}s", state="complete", expanded=False)
                    
                    if user_edited_sql.strip() != st.session_state.original_generated_sql.strip():
                        st.session_state.pending_feedback = True
                        st.session_state.edited_sql_for_feedback = user_edited_sql
                    else:
                        st.session_state.pending_feedback = False
                        
                except Exception as e:
                    status.update(label="❌ Query Error", state="error")
                    st.error(f"Execution failed: {e}")

    with col_res:
        st.subheader("📊 Data Results")
        
        if st.session_state.get("pending_feedback", False):
            st.info("💡 **Looks like you have modified the LLM query. Do you want to retrain the model with these changes?**")
            btn_col1, btn_col2, _ = st.columns([1.5, 1.5, 4]) 
            
            with btn_col1:
                if st.button("🧠 Yes, retrain the model", use_container_width=True):
                    with st.spinner("Analyzing and updating ontology..."):
                        evaluate_and_update_ontology(
                            user_input, 
                            st.session_state.original_generated_sql, 
                            st.session_state.edited_sql_for_feedback
                        )
                    st.session_state.pending_feedback = False
                    st.rerun()
                    
            with btn_col2:
                if st.button("❌ No, ignore", type="secondary", use_container_width=True):
                    st.session_state.pending_feedback = False
                    st.rerun()

        if "last_df" in st.session_state:
            st.dataframe(st.session_state.last_df, use_container_width=True)

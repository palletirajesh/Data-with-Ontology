import os
import time
import json
from datetime import date
import streamlit as st
import pandas as pd
import requests
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

# --- 2. ENGINES ---
# IMPORTANT: @st.cache_resource removed here to prevent stale connection hangs
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

def evaluate_and_update_ontology(user_text, original_sql, edited_sql):
    evaluation_prompt = f"""
    The user asked: "{user_text}"
    The AI generated: "{original_sql}"
    The User corrected to: "{edited_sql}"
    
    Did the user fix a table join, column mapping, or business jargon? 
    If YES, extract the new mapping as a valid JSON array of objects (using our bank ontology).
    If NO, return exactly "MISMATCH".
    """
    
    response = call_groq_llm(evaluation_prompt) 
    
    if "MISMATCH" not in response:
        try:
            clean_json_str = response.replace("```json", "").replace("```", "").strip()
            new_knowledge_json = json.loads(clean_json_str)
            
            if isinstance(new_knowledge_json, dict):
                new_knowledge_json = [new_knowledge_json]
                
            today_str = date.today().isoformat()
            for item in new_knowledge_json:
                item["dateAdded"] = today_str
                item["reviewStatus"] = "Pending_Review"
            
            g = rdflib.Graph()
            g.parse("knowledge_base.jsonld", format="json-ld")
            new_graph = rdflib.Graph().parse(data=json.dumps(new_knowledge_json), format="json-ld")
            g += new_graph
            g.serialize(destination="knowledge_base.jsonld", format="json-ld")
            
            st.success(f"✅ Model successfully retrained! Changes flagged for review on {today_str}.")
            
        except json.JSONDecodeError:
            st.warning("Feedback loop skipped: AI did not return strictly valid JSON.")
        except Exception as e:
            st.error(f"Error updating ontology: {e}")

# --- 4. SEMANTIC CONTEXT LOADER ---
def build_context_string():
    context = ""
    try:
        with open("database_schema.md", "r") as f:
            context += f"--- DATABASE SCHEMA ---\n{f.read()}\n\n"
    except Exception as e:
        st.warning(f"Could not load schema: {e}")

    try:
        with open("knowledge_base.jsonld", "r") as f:
            context += f"--- ONTOLOGY & JARGON MAPPING ---\n{f.read()}\n"
    except Exception as e:
        st.warning(f"Could not load ontology: {e}")
        
    return context

# --- 5. MAIN APP UI ---
st.title("🏦 Risk Data Agent")
st.markdown("Ask natural language questions to generate BigQuery SQL logic.")

user_input = st.text_input("What risk data do you need?", placeholder="e.g., Show me Amazon customers with late payments...")

if user_input:
    col_sql, col_res = st.columns([1, 1.5])
    
    with col_sql:
        st.subheader("⚙️ Generated Logic")
        
        # --- THE FIX: We ONLY call Groq if the user asked a BRAND NEW question! ---
        is_new_question = ("last_user_input" not in st.session_state or st.session_state.last_user_input != user_input)
        
        if is_new_question:
            with st.spinner("Translating intent to SQL via Llama-3..."):
                knowledge_context = build_context_string()
                full_path = f"{BQ_PROJECT}.{BQ_DATASET}"
                
                # All 19 Rules Kept Safely Intact
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
                    "4. DYNAMIC COLUMNS: In addition to the default columns, you MUST select the columns that relate to the user's conditions (e.g., if they ask about scores and dues, select `fico_score`, `payment_due_amount`, and `days_past_due`).\n"
                    "5. PROACTIVE JOINS: If picking dynamic columns requires other tables (like `card_partner` from dim_card_association), you MUST write the JOIN for that table.\n"
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
                    "17. TRANSPARENCY RULE (CRITICAL): Any column you use in the WHERE or HAVING clause MUST also be included in the SELECT clause. If you filter by a column, the user must be able to see it to verify your math (e.g., if you filter by `actual_payment_made`, you MUST SELECT `actual_payment_made`).\n"
                    "18. GRANULARITY: Unless the user explicitly uses words like 'count', 'total', or 'how many', ALWAYS return a detailed list of records (SELECT explicit columns) rather than a summary or count.\n"
                    "19. ALIASES: Whenever writing a query that contains a JOIN, you must assign short table aliases (e.g., t1, t2) and explicitly prefix every single column in the SELECT and WHERE clauses with its corresponding alias. Never leave a column name unqualified.\n\n"
                    f"Write BigQuery SQL for this user request: \"{user_input}\""
                )
                
                final_sql = call_groq_llm(system_prompt).replace("```sql", "").replace("```", "").strip()
                
                # Save state so we don't call Groq again unless the user text changes
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
                    # THE FIX: Added timeout=30 and create_bqstorage_client=False to bypass local gRPC network drops
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

# --- 6. ARCHITECTURE DETAILS ---
st.divider()
st.subheader("🏗️ System Architecture")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.info("**1. User Interface**\nStreamlit collects natural language inputs and provides an editable SQL form.")
with c2:
    st.info("**2. Semantic Layer**\nOWL JSON-LD Ontology maps Jargon to implicit database schemas & aliases.")
with c3:
    st.info("**3. Logic Synthesis**\n**Llama-3.3-70B** via Groq generates BigQuery SQL from semantic chunks.")
with c4:
    st.info("**4. Feedback Loop**\nAgent evaluates user SQL edits and automatically appends learned JSON-LD mapping.")

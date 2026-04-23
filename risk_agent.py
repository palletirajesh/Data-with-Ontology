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

# --- 2. ENGINES ---
@st.cache_resource
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
        "temperature": 0.1
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        st.error(f"LLM Error: {response.text}")
        return ""

def evaluate_and_update_ontology(user_text, original_sql, edited_sql):
    """Evaluates user SQL edits, adds tracking metadata, and safely updates JSON-LD."""
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
            
            st.toast(f"🔄 Learned new mapping. Flagged for review on {today_str}.")
            
            g = rdflib.Graph()
            g.parse("knowledge_base_23Ap.jsonld", format="json-ld")
            new_graph = rdflib.Graph().parse(data=json.dumps(new_knowledge_json), format="json-ld")
            g += new_graph
            g.serialize(destination="knowledge_base_23Ap.jsonld", format="json-ld")
            
        except json.JSONDecodeError:
            st.warning("Feedback loop skipped: AI did not return strictly valid JSON.")
        except Exception as e:
            st.error(f"Error updating ontology: {e}")

# --- 4. SEMANTIC CONTEXT LOADER (Restored from Original/v0) ---
def build_context_string():
    """Loads the explicit schema and JSON-LD mapping to ground the LLM"""
    context = ""
    # 1. Load Markdown Schema
    try:
        with open("database_schema_23ap.md", "r") as f:
            context += f"--- DATABASE SCHEMA ---\n{f.read()}\n\n"
    except Exception as e:
        st.warning(f"Could not load schema: {e}")

    # 2. Load JSON-LD Ontology
    try:
        with open("knowledge_base_23Ap.jsonld", "r") as f:
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
        
        with st.spinner("Translating intent to SQL via Llama-3..."):
            
            # --- RESTORED LOGIC: Injecting the actual knowledge base context ---
            knowledge_context = build_context_string()
            
            # Combine the strict alias rules + the context + the user prompt
            system_prompt = f"""
            You are a BigQuery SQL expert. Use the SCHEMA and ONTOLOGY provided below to write a query.
            
            CRITICAL SQL RULE: Whenever writing a query with a JOIN, you MUST assign short 
            table aliases (e.g., t1, t2) and explicitly prefix EVERY SINGLE column in the 
            SELECT and WHERE clauses with its corresponding alias to prevent ambiguous columns.
            
            {knowledge_context}
            
            Write BigQuery SQL for this user request: "{user_input}"
            Return ONLY the raw SQL code, nothing else. No markdown formatting.
            """
            
            final_sql = call_groq_llm(system_prompt).replace("```sql", "").replace("```", "").strip()
            
        if "last_user_input" not in st.session_state or st.session_state.last_user_input != user_input:
            st.session_state.last_user_input = user_input
            st.session_state.sql_editor_key = final_sql 
            st.session_state.original_generated_sql = final_sql
            if "last_df" in st.session_state:
                del st.session_state["last_df"]

        st.subheader("📝 Review and Edit SQL")
        
        # --- WORKING EXECUTION BLOCK (From v1) ---
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
                    df = bq_client.query(user_edited_sql).to_dataframe()
                    end_time = time.perf_counter()
                    st.session_state.last_df = df
                    status.update(label=f"✅ Query Finished in {end_time - start_time:.2f}s", state="complete", expanded=False)
                    
                    if user_edited_sql.strip() != st.session_state.original_generated_sql.strip():
                        evaluate_and_update_ontology(
                            user_input, 
                            st.session_state.original_generated_sql, 
                            user_edited_sql
                        )
                        
                except Exception as e:
                    status.update(label="❌ Query Error", state="error")
                    st.error(e)

    with col_res:
        st.subheader("📊 Data Results")
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

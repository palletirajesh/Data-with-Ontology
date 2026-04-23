import os
import time
import streamlit as st
import pandas as pd
import requests
import rdflib
from google.cloud import bigquery
from google.oauth2 import service_account
from sentence_transformers import SentenceTransformer, util
import json
from datetime import date


# --- 1. CONFIG & STYLE ---
st.set_page_config(page_title="Risk Data Agent", page_icon="🏦", layout="wide")

# Professional Fonts and CSS
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

# --- 3. KNOWLEDGE & SEMANTIC LOGIC ---
@st.cache_data
def get_unified_knowledge():
    knowledge_chunks = []
    try:
        with open("database_schema.md", "r") as f:
            content = f.read()
            knowledge_chunks.extend(["TABLE_STRUCT: " + t for t in content.split("### TABLE:")[1:]])
    except: pass

    try:
        g = rdflib.Graph().parse("knowledge_base.jsonld", format="json-ld")
        q_joins = "SELECT ?tLabel ?sK ?tK WHERE { ?j a <http://com/ontology#JoinDefinition> ; <http://com/ontology#targetTable> ?t ; <http://com/ontology#sourceKey> ?sK ; <http://com/ontology#targetKey> ?tK . ?t <http://www.w3.org/2000/01/rdf-schema#label> ?tLabel . }"
        for row in g.query(q_joins):
            knowledge_chunks.append(f"MANDATORY JOIN: Join to {row.tLabel} ON {row.sK} = {row.tK}")
        
        q_jargon = "SELECT ?colLabel ?jargon WHERE { ?col <http://www.w3.org/2000/01/rdf-schema#label> ?colLabel ; <http://com/ontology#businessJargon> ?jargon . }"
        for row in g.query(q_jargon):
            knowledge_chunks.append(f"BUSINESS TRANSLATION: Term '{row.jargon}' maps to column '{row.colLabel}'")
    except: pass
    return knowledge_chunks

def get_semantic_context(query, knowledge_base, top_k=8):
    query_emb = embedder.encode(f"search_query: {query}", convert_to_tensor=True)
    kb_embs = embedder.encode(knowledge_base, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, kb_embs, top_k=top_k)
    return "\n\n".join([knowledge_base[hit['corpus_id']] for hit in hits[0]])

# --- 4. DATA DOWNLOADER ---
@st.cache_data
def get_full_dataset_csv():
    # Pulls a small sample so users can understand the DB structure
    query = f"SELECT * FROM `{BQ_PROJECT}.{BQ_DATASET}.dim_customer` LIMIT 100"
    return bq_client.query(query).to_dataframe().to_csv(index=False)

# --- 5. INTELLIGENT SQL GENERATION ---
def generate_sql(user_query, context):
    full_path = f"{BQ_PROJECT}.{BQ_DATASET}"
    
    system_prompt = f"""You are a BigQuery SQL Expert. 
    Context: {context}
    
    STRICT RULES:
    1. ONLY use Tables/Columns in Context.
    2. NEVER USE 'SELECT *'. You must explicitly name the columns in your SELECT statement.
    3. DEFAULT COLUMNS: Whenever a user asks about 'customers' or 'clients', you MUST ALWAYS select at least:
       - `cust_id`
       - `customer_name`
       - `card_id`
    4. DYNAMIC COLUMNS: In addition to the default columns, you MUST select the columns that relate to the user's conditions (e.g., if they ask about scores and dues, select `fico_score`, `payment_due_amount`, and `days_past_due`).
    5. PROACTIVE JOINS: If picking dynamic columns requires other tables (like `card_partner` from dim_card_association), you MUST write the JOIN for that table.
    6. Table Format: ALWAYS use backticks and the full path: `{full_path}.table_name`
    7. NO PARENTHESES: Never put () after a table name.
    8. Use MANDATORY JOINs exactly. Sequence tables logically.
    9. Output ONLY raw SQL code. No markdown or explanations.
    10. If data is missing, output EXACTLY: "I cannot answer this with the available data."
    11. Always begin with a SELECT clause. For multi-table joins, dim_customer should act a bridge table
    12. EVERY table name in the SQL must be prefixed with: `{full_path}.
    13. Example format: SELECT * FROM `{full_path}.dim_customer` JOIN `{full_path}.dim_card_association`
    14. Tables available: dim_customer, dim_card_association, fact_card_ledger, fact_credit_bureau.
    15. For string filters, use: UPPER(column) = UPPER('value').
    16. TRANSLATION: Apply BUSINESS TRANSLATION RULES strictly to map user jargon to correct columns.
    17. TRANSPARENCY RULE (CRITICAL): Any column you use in the WHERE or HAVING clause MUST also be included in the SELECT clause. If you filter by a column, the user must be able to see it to verify your math (e.g., if you filter by `actual_payment_made`, you MUST SELECT `actual_payment_made`).
    18. GRANULARITY: Unless the user explicitly uses words like 'count', 'total', or 'how many', ALWAYS return a detailed list of records (SELECT *) rather than a summary or count.
    """
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": "llama-3.3-70b-versatile", "messages": messages, "temperature": 0}
    ).json()
    
    return response['choices'][0]['message']['content'].strip().replace("```sql", "").replace("```", "").strip()
    # 1. Store the generated SQL in session state so it doesn't reset on interaction
    if "current_sql" not in st.session_state:
        st.session_state.current_sql = final_sql # From your Llama-3 output
    if "original_prompt" not in st.session_state:
        st.session_state.original_prompt = user_input # The original text prompt

    st.subheader("📝 Review and Edit SQL")
# 2. Provide a text area for the user to edit the query
    user_edited_sql = st.text_area(
        "Modify the query below if needed before running:", 
        value=st.session_state.current_sql, 
        height=250
    )

# 3. Use a form or a button to trigger execution of the EDITED query
    if st.button("▶️ Execute Query"):
        start_time = time.perf_counter()
        with st.status("🚀 Running BigQuery Job...", expanded=True) as status:
            try:
            # RUN THE EDITED SQL, NOT THE ORIGINAL
                df = bq_client.query(user_edited_sql).to_dataframe()
                end_time = time.perf_counter()
                st.session_state.last_df = df
                status.update(label=f"✅ Query Finished in {end_time - start_time:.2f}s", state="complete", expanded=False)
            
                # 4. TRIGGER THE FEEDBACK LOOP IF THE QUERY WAS CHANGED
                if user_edited_sql.strip() != st.session_state.current_sql.strip():
                    evaluate_and_update_ontology(st.session_state.original_prompt, st.session_state.current_sql, user_edited_sql)
                
            except Exception as e:
                status.update(label="❌ Query Error", state="error")
                st.error(e)
    
def get_client_location():
    """Silently fetches IP and location data for the admin log."""
    try:
        # Uses a free, no-auth IP geolocation API
        response = requests.get('http://ip-api.com/json/').json()
        return {
            "ip": response.get("query", "Unknown"),
            "city": response.get("city", "Unknown"),
            "state": response.get("regionName", "Unknown"),
            "zip": response.get("zip", "Unknown")
        }
    except:
        return {"ip": "Unknown", "city": "Unknown", "state": "Unknown", "zip": "Unknown"}
# --- 6. SIDEBAR HISTORY ---
# --- UPDATE YOUR SIDEBAR SECTION ---
with st.sidebar:
    st.title("🕒 Query History")
    try:
        # BigQuery handles the timezone conversion to EST right in the SQL!
        hist_query = f"""
            SELECT 
                user_query, 
                FORMAT_TIMESTAMP('%b %d, %I:%M %p', created_at, 'America/New_York') as est_time 
            FROM `{BQ_PROJECT}.{BQ_DATASET}.query_history` 
            ORDER BY created_at DESC 
            LIMIT 10
        """
        hist_df = bq_client.query(hist_query).to_dataframe()
        
        # Displaying only the Query and the EST Time in the UI
        for index, row in hist_df.iterrows():
            st.markdown(f"**🗨️ {row['user_query']}**")
            st.caption(f"⏱️ {row['est_time']} EST")
            st.divider()
    except Exception as e:
        st.write("No history available yet.")
        
def evaluate_and_update_ontology(user_text, original_sql, edited_sql):
    """Evaluates user SQL edits, adds tracking metadata, and updates JSON-LD."""
    
    evaluation_prompt = f"""
    The user asked: "{user_text}"
    The AI generated: "{original_sql}"
    The User corrected to: "{edited_sql}"
    
    Did the user fix a table join, column mapping, or business jargon? 
    If YES, extract the new mapping as a valid JSON array of objects (using our bank ontology).
    If NO, return exactly "MISMATCH".
    """
    
    # Call Groq (Llama-3)
    response = call_groq_llm(evaluation_prompt) 
    
    if "MISMATCH" not in response:
        try:
            # Parse the LLM's JSON response
            new_knowledge_json = json.loads(response)
            
            # Ensure it's a list for iteration
            if isinstance(new_knowledge_json, dict):
                new_knowledge_json = [new_knowledge_json]
                
            # --- INJECT TRACKING METADATA ---
            today_str = date.today().isoformat()
            for item in new_knowledge_json:
                item["dateAdded"] = today_str
                item["reviewStatus"] = "Pending_Review"
            # --------------------------------
            
            st.toast(f"🔄 Learned new mapping. Flagged for review on {today_str}.")
            
            # Convert back to string and merge into rdflib Graph
            g = rdflib.Graph()
            g.parse("knowledge_base_23Ap.jsonld", format="json-ld")
            
            new_graph = rdflib.Graph().parse(data=json.dumps(new_knowledge_json), format="json-ld")
            g += new_graph
            
            # Save the updated file
            g.serialize(destination="knowledge_base_23Ap.jsonld", format="json-ld")
            
        except json.JSONDecodeError:
            print("Error: LLM did not return valid JSON.")
        
# --- 7. MAIN UI ---
st.title("🏦 Risk Data Intelligence Agent")

# Requirement 2: Small download button above input
col_dl, _ = st.columns([1, 3])
with col_dl:
    st.download_button(
        label="📥 Download Data Sample",
        data=get_full_dataset_csv(),
        file_name="additional_data.csv",
        mime="text/csv",
    )

user_input = st.text_input("Analyze your risk data:", placeholder="e.g. Names of clients with FICO scores > 750")

if user_input:
    # Requirement 6: Clear old results by not displaying session_state until button click
    kb = get_unified_knowledge()
    context = get_semantic_context(user_input, kb)
    generated_sql = generate_sql(user_input, context)

    col_sql, col_res = st.columns([4, 6])
    
    with col_sql:
        st.subheader("📝 Generated SQL")
        final_sql = st.text_area("Review AI Logic:", value=generated_sql, height=250)
        
        # Requirement 5: Execution state and timer
        if st.button("▶️ Execute Query"):
            start_time = time.perf_counter()
            with st.status("🚀 Running BigQuery Job...", expanded=True) as status:
                try:
                    df = bq_client.query(final_sql).to_dataframe()
                    end_time = time.perf_counter()
                    st.session_state.last_df = df
                    status.update(label=f"✅ Query Finished in {end_time - start_time:.2f}s", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="❌ Query Error", state="error")
                    st.error(e)

    with col_res:
        st.subheader("📊 Data Results")
        if "last_df" in st.session_state:
            st.dataframe(st.session_state.last_df, use_container_width=True)

# Requirement 4: Expanded Architecture Details
st.divider()
st.subheader("🏗️ System Architecture")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.info("**1. User Interface**\nStreamlit collects natural language inputs and provides contextual data downloads.")
with c2:
    st.info("**2. Semantic Layer**\nInput is passed through **MiniLM Embeddings** to map jargon against the **JSON-LD Ontology**.")
with c3:
    st.info("**3. Logic Synthesis**\n**Llama-3.3-70B** reads the combined schema/ontology to engineer BigQuery-compliant SQL.")
with c4:
    st.info("**4. Cloud Warehouse**\n**Google BigQuery** executes serverless queries, returning production-grade results to the UI.")

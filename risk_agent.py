import streamlit as st
import pandas as pd
import requests
import re
import time
import json
import rdflib
import os
from databricks import sql

# ==========================================
# --- CONFIGURATION (PASTE YOUR KEY HERE) ---
# ==========================================
# Make sure to replace this with your actual Groq API Key!
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

DB_CONFIG = {
    "server_hostname": "dbc-77f699f6-0373.cloud.databricks.com", 
    "http_path": "/sql/1.0/warehouses/15171b6cf0a70633", 
    "access_token": "dapide1a7848b6a9ec77bd403ebd3ece1672"
}

# --- 1. CORE CONNECTIONS ---
@st.cache_resource
def get_db_connection():
    return sql.connect(**DB_CONFIG)

conn = get_db_connection()

@st.cache_resource
def load_ontology():
    """Loads the RDF/OWL Graph into memory"""
    g = rdflib.Graph()
    try:
        g.parse("knowledge_base.jsonld", format="json-ld")
    except Exception as e:
        st.error(f"Failed to load Ontology: {e}")
    return g

# --- 2. ONTOLOGY ENGINE ---
def get_ontology_context(user_query, g):
    """Multi-Hop Reasoning: Jargon -> Concept -> Column -> Table -> Explicit Joins"""
    user_query_lower = user_query.lower()
    matched_concepts = set()
    bank = rdflib.Namespace("http://gen_ai_bank.com/ontology#")
    
    # Hop 1: Find matched business jargon in the graph
    for concept, _, jargon in g.triples((None, bank.businessJargon, None)):
        if str(jargon).lower() in user_query_lower:
            matched_concepts.add(concept)
            
    if not matched_concepts:
        return "ERROR: No semantic concepts recognized in the query."

    # Hop 2 & 3: SPARQL Traversal to find Tables, Columns, AND Explicit Joins
    values_clause = " ".join([f"<{str(c)}>" for c in matched_concepts])
    
    sparql_query = f"""
    PREFIX bank: <http://gen_ai_bank.com/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT DISTINCT ?tableLabel ?colLabel ?targetTableLabel ?sourceKey ?targetKey
    WHERE {{
        VALUES ?concept {{ {values_clause} }}
        ?col bank:representsConcept ?concept .
        ?table bank:hasColumn ?col .
        ?table rdfs:label ?tableLabel .
        ?col rdfs:label ?colLabel .
        
        # Optional: Pull explicit join definitions if this table has them
        OPTIONAL {{
            ?table bank:joinsWith ?joinObj .
            ?joinObj bank:targetTable ?tTable .
            ?tTable rdfs:label ?targetTableLabel .
            ?joinObj bank:sourceKey ?sourceKey .
            ?joinObj bank:targetKey ?targetKey .
        }}
    }}
    """
    
    results = g.query(sparql_query)
    
    # Format the exact logical schema for the LLM
    schemas = {}
    joins = set()
    
    for row in results:
        t_name = str(row.tableLabel)
        c_name = str(row.colLabel)
        
        if t_name not in schemas:
            schemas[t_name] = set()
        schemas[t_name].add(c_name)
        
        # If the SPARQL query found an explicit join rule, save it
        if row.targetTableLabel:
            target_name = str(row.targetTableLabel)
            s_key = str(row.sourceKey)
            t_key = str(row.targetKey)
            
            join_rule = f"JOIN {t_name} TO {target_name} ON {s_key} = {t_key}"
            joins.add(join_rule)
            
            # THE FIX: Automatically grant the LLM permission to use the join keys!
            schemas[t_name].add(s_key)
            if target_name not in schemas:
                schemas[target_name] = set()
            schemas[target_name].add(t_key)
            
    # --- THESE ARE THE LINES THAT GOT DELETED ---
    context = "ONTOLOGY SCHEMA MATCHES:\n"
    for table, cols in schemas.items():
        context += f"TABLE: {table}\nREQUIRED COLUMNS: {', '.join(cols)}\n\n"
        
    if joins:
        context += "MANDATORY JOIN RULES:\n"
        for j in joins:
            context += f"- {j}\n"
            
    return context.strip()

# --- 3. SQL GENERATION ---
def generate_sql(user_query, context):
    prompt = f"""
    SYSTEM: You are a Databricks SQL Expert powered by an RDF Ontology.
    
    {context}
    
    STRICT RULES:
    1. ONLY use the Tables and Columns explicitly provided in the ONTOLOGY SCHEMA MATCHES above.
    2. If MANDATORY JOIN RULES are provided, you MUST use them exactly as written. If no join rules are provided but multiple tables exist, join them logically.
    3. Output ONLY the raw SQL code. No markdown, no explanations.
    4. SEQUENCE YOUR JOINS LOGICALLY. You cannot reference a table alias in an ON clause until that table has been introduced. If dim_customer is provided as a Bridge Table, start your FROM clause there.
    5. If the user asks for a column or metric that cannot be calculated from the given columns, output exactly: "I cannot answer this with the available data."
    6. TRANSLATION: Do NOT copy the user's words into the SQL. If the user asks for "payment delay", you MUST translate that to the exact column 'days_past_due'.
    7. Output ONLY the raw SQL code. No markdown, no explanations.
    8. For the string datatype, convert any case to Upper Case like upper(data_unit) = upper("Amazon')
    
    QUESTION: {user_query}
    SQL:"""
    
    start = time.time()
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
    
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload).json()
    latency = time.time() - start
    
    if 'error' in response: 
        return f"API ERROR: {response['error']['message']}", latency
        
    raw_sql = response['choices'][0]['message']['content']
    clean_sql = re.sub(r'```sql|```', '', raw_sql).strip()
    return clean_sql.split(";")[0] + ";", latency

# --- 4. GRAPH META-AGENT (Self-Healing Ontology) ---
def auto_update_ontology(user_query, bad_sql, good_sql):
    with open("knowledge_base.jsonld", "r") as f:
        kb = json.load(f)
        
    # Get all valid OWL Classes (Concepts) from the JSON-LD
    valid_concepts = [node["@id"] for node in kb["@graph"] if node.get("@type") == "owl:Class"]
        
    prompt = f"""
    SYSTEM: You are a Semantic Web Administrator managing an OWL Ontology.
    User asked: "{user_query}"
    AI generated: {bad_sql}
    User fixed it to: {good_sql}
    
    Identify the semantic concept the user had to map their jargon to.
    
    RULES:
    1. You MUST choose EXACTLY ONE valid ID from this list: {valid_concepts}
    
    Return ONLY a valid JSON object matching this structure. NO markdown:
    {{"jargon": "the raw word user typed", "concept_id": "valid_id_here"}}
    """
    
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload).json()
        raw_resp = re.sub(r'```json|```', '', response['choices'][0]['message']['content']).strip()
        new_hook = json.loads(raw_resp)
        
        concept_id = new_hook.get("concept_id", "")
        jargon = new_hook.get("jargon", "")
        
        if concept_id not in valid_concepts:
            return False, f"Agent hallucinated concept: {concept_id}"
            
        # Update the JSON-LD Graph natively
        for node in kb["@graph"]:
            if node.get("@id") == concept_id:
                if "businessJargon" not in node:
                    node["businessJargon"] = []
                if jargon not in node["businessJargon"]:
                    node["businessJargon"].append(jargon)
                break
                
        with open("knowledge_base.jsonld", "w") as f:
            json.dump(kb, f, indent=2)
            
        # Clear Streamlit cache so the new graph is loaded on next run
        st.cache_resource.clear()
            
        return True, f"Mapped '{jargon}' to {concept_id}"
        
    except Exception as e:
        return False, f"System Error: {str(e)}"

# --- 5. INTERACTIVE UI ---
st.set_page_config(layout="wide", page_title="GenAI Graph Agent")
st.title("🏦 GenAI Bank Agent (Human-in-the-Loop 🕸️)")

g = load_ontology()

# State Management
if "current_sql" not in st.session_state: st.session_state.current_sql = ""
if "original_sql" not in st.session_state: st.session_state.original_sql = ""
if "last_query" not in st.session_state: st.session_state.last_query = ""
if "df" not in st.session_state: st.session_state.df = None
if "gen_time" not in st.session_state: st.session_state.gen_time = 0.0

# 1. The Input
user_query = st.text_input("Ask a question about customer risk (Press Enter to Generate):", placeholder="e.g., Show Amazon customers with a payment delay")

# 2. The Generation Step
if user_query and user_query != st.session_state.last_query:
    st.session_state.last_query = user_query 
    st.session_state.df = None # Clear old data
    
    with st.spinner("🧠 Traversing Ontology & Generating SQL..."):
        context = get_ontology_context(user_query, g)
        if "ERROR" in context:
            st.error(context)
            st.session_state.current_sql = ""
        else:
            st.session_state.current_sql, st.session_state.gen_time = generate_sql(user_query, context)
            st.session_state.original_sql = st.session_state.current_sql

# 3. The Review & Execute Step
if st.session_state.current_sql:
    st.subheader("📜 Review AI-Generated SQL")
    
    # Editable text area showing the SQL
    edited_sql = st.text_area("You can edit the query below before executing:", value=st.session_state.current_sql, height=150)
    
    col_run, col_learn = st.columns(2)
    
    # EXECUTION BUTTON
    with col_run:
        if st.button("▶️ Execute Query on Databricks", type="primary"):
            with st.spinner("⚡ Executing on Databricks..."):
                try:
                    exec_start = time.time()
                    st.session_state.df = pd.read_sql(edited_sql, conn)
                    st.session_state.current_sql = edited_sql # Save the exact code that was run
                    st.success(f"✅ Data retrieved in {round(time.time() - exec_start, 2)}s")
                except Exception as e:
                    st.error(f"❌ SQL Execution Error: {str(e)}")
                    st.session_state.df = None
                    
    # LEARNING BUTTON (Only appears if the code was manually changed)
    with col_learn:
        if edited_sql.strip() != st.session_state.original_sql.strip():
            st.warning("⚠️ You modified the AI's query.")
            if st.button("🕸️ Teach AI your edits (Update Ontology)"):
                with st.spinner("Wiring new semantic edges..."):
                    success, result = auto_update_ontology(st.session_state.last_query, st.session_state.original_sql, edited_sql)
                    if success:
                        st.success(result)
                        st.session_state.original_sql = edited_sql # Reset to hide button
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error(f"Failed to learn: {result}")

# 4. The Results Display
if st.session_state.df is not None:
    st.markdown("---")
    st.subheader("📊 Databricks Results")
    st.dataframe(st.session_state.df, use_container_width=True)
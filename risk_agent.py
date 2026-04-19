import streamlit as st
import pandas as pd
import requests
import rdflib
import io
from databricks import sql
from sentence_transformers import SentenceTransformer, util

# ==========================================
# --- 1. CONFIGURATION & SECRETS ---
# ==========================================
st.set_page_config(page_title="Risk Data Agent", page_icon="🏦", layout="wide")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
DB_CONFIG = {
    "server_hostname": st.secrets["DB_SERVER_HOSTNAME"], 
    "http_path": st.secrets["DB_HTTP_PATH"], 
    "access_token": st.secrets["DB_ACCESS_TOKEN"]
}

# --- 1. CONFIGURATION & SECRETS ---
SOURCE_FILES = {
    "additional_data.xlsx - dim_customer.csv": "dim_customer",
    "additional_data.xlsx - fact_card_ledger.csv": "fact_card_ledger",
    "additional_data.xlsx - dim_card_association.csv": "dim_card_association",
    "additional_data.xlsx - fact_credit_bureau.csv": "fact_credit_bureau"
}

# ==========================================
# --- 2. CORE ENGINES (NOMIC & DB) ---
# ==========================================
@st.cache_resource
def load_nomic_model():
    """High-res embeddings that understand 'Client' == 'Customer' natively."""
    return SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)

embedder = load_nomic_model()

@st.cache_resource
def get_db_connection():
    return sql.connect(**DB_CONFIG)

conn = get_db_connection()

# --- INVISIBLE TELEMETRY ENGINE ---
@st.cache_data(ttl=3600)
def get_client_location():
    """Silently captures City and Country based on browser headers."""
    try:
        # Streamlit >= 1.37 passes the user's IP in the headers
        headers = st.context.headers
        ip = headers.get("X-Forwarded-For", "").split(",")[0].strip()
        
        if not ip or ip == "127.0.0.1" or ip == "localhost":
            return "Local Network"
            
        # Free API to convert IP to City/Country
        response = requests.get(f"http://ip-api.com/json/{ip}", timeout=2)
        data = response.json()
        
        if data.get("status") == "success":
            return f"{data['city']}, {data['country']}"
        return "Unknown Location"
    except Exception:
        return "Unknown Location"

# ==========================================
# --- 3. GLOBAL SLIDING WINDOW (MEMORY) ---
# ==========================================
def init_global_history():
    """Verifies that the existing history table is accessible."""
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM gen_ai_bank.query_history LIMIT 1")
    except Exception:
        st.sidebar.error("⚠️ Global History Table not accessible in Databricks.")

def save_to_global_history(query, sql_code):
    location = get_client_location()
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO gen_ai_bank.query_history (user_query, generated_sql, user_location) VALUES (?, ?, ?)", 
            [query, sql_code, location]
        )

def load_global_history(limit=10):
    try:
        return pd.read_sql(f"SELECT user_location, user_query as query, generated_sql as sql FROM gen_ai_bank.query_history ORDER BY created_at DESC LIMIT {limit}", conn).to_dict('records')
    except: return []

# ==========================================
# --- 4. UNIFIED CONTEXT ENGINE ---
# ==========================================
@st.cache_data
def get_unified_knowledge():
    knowledge_chunks = []
    # 1. Load technical schema from MD
    with open("database_schema.md", "r") as f:
        content = f.read()
        knowledge_chunks.extend(["TABLE_STRUCT: " + t for t in content.split("### TABLE:")[1:]])
    
    # 2. Load mandatory Join Rules from JSONLD
    g = rdflib.Graph().parse("knowledge_base.jsonld", format="json-ld")
    q_joins = """
    SELECT ?tLabel ?sK ?tK WHERE { 
        ?j a <http://gen_ai_bank.com/ontology#JoinDefinition> ; 
           <http://gen_ai_bank.com/ontology#targetTable> ?t ; 
           <http://gen_ai_bank.com/ontology#sourceKey> ?sK ; 
           <http://gen_ai_bank.com/ontology#targetKey> ?tK . 
        ?t rdfs:label ?tLabel . 
    }"""
    for row in g.query(q_joins):
        knowledge_chunks.append(f"MANDATORY JOIN: Join to {row.tLabel} ON {row.sK} = {row.tK}")
            
    return knowledge_chunks

def get_semantic_context(query, knowledge_base, top_k=10):
    query_emb = embedder.encode(f"search_query: {query}", convert_to_tensor=True)
    kb_embs = embedder.encode(knowledge_base, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, kb_embs, top_k=top_k)
    return "\n\n".join([knowledge_base[hit['corpus_id']] for hit in hits[0]])

# ==========================================
# --- 5. THE SQL ENGINEER (GROQ LLAMA-3.3) ---
# ==========================================
def generate_sql(user_query, context, history):
    
    system_prompt = f"""You are a Databricks SQL Expert. Context:
    {context}
    
    STRICT RULES:
    1. ONLY use Tables/Columns in Context.
    2. Use MANDATORY JOINs exactly. Sequence tables logically.
    3. Output ONLY raw SQL code. No markdown or explanations.
    4. If data is missing, output EXACTLY: "I cannot answer this with the available data."
    5. Start FROM 'gen_ai_bank.dim_customer' for multi-table joins.
    6. For string filters, use: UPPER(column) = UPPER('value').
    7. TRANSLATION: Map terms like 'Amazon' to card_partner = 'AMAZON'.
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-3:]:
        messages.append({"role": "user", "content": h['query']})
        messages.append({"role": "assistant", "content": h['sql']})
    messages.append({"role": "user", "content": user_query})

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": "llama-3.3-70b-versatile", "messages": messages, "temperature": 0.1}
    )
    
    res = response.json()
    if "choices" in res:
        sql_out = res['choices'][0]['message']['content'].strip()
        return sql_out.replace("```sql", "").replace("```", "").strip()
    return "I cannot answer this with the available data."

# ==========================================
# --- 6. UI: HEADER & DOWNLOAD ---
# ==========================================
st.title("🏦 Risk Data Agent")

col_head, col_dl = st.columns([0.8, 0.2])
with col_head:
    user_input = st.text_input("Query Bank Data (Shared Memory):", placeholder="e.g. Clients with FICO > 700 at Amazon")

with col_dl:
    # Direct download of the single Excel file
    try:
        with open("additional_data.xlsx", "rb") as f:
            st.download_button(
                label="Download DB",
                data=f,
                file_name="bank_risk_database.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                icon="💾",
                help="Download the underlying Excel database."
            )
    except FileNotFoundError:
        st.error("Excel file not found.")
# ==========================================
# --- 7. MAIN EXECUTION ---
# ==========================================
init_global_history()
kb = get_unified_knowledge()
history = load_global_history()

if user_input:
    context = get_semantic_context(user_input, kb)
    generated_sql = generate_sql(user_input, context, history)
    
    col_l, col_r = st.columns([4, 6])
    with col_l:
        st.markdown("### 📝 SQL Draft")
        final_sql = st.text_area("SQL:", value=generated_sql, height=250)
        if st.button("▶️ Run Query (user can modify as well)", type="primary", use_container_width=True):
            try:
                st.session_state.df = pd.read_sql(final_sql, conn)
                save_to_global_history(user_input, final_sql)
                st.rerun()
            except Exception as e: st.error(f"SQL Error: {e}")

    with col_r:
        st.markdown("### 📊 Results")
        if "df" in st.session_state:
            st.dataframe(st.session_state.df, use_container_width=True)

# ==========================================
# --- 8. TECHNICAL RATIONALE ---
# ==========================================
st.divider()
st.subheader("🛠️ Technical Architecture & Rationale")
t1, t2, t3, t4 = st.columns(4)
with t1:
    st.markdown("#### 🧠 Llama-3.3-70B")
    st.caption("**SQL Engineering**")
    st.write("Ensures complex JOIN accuracy and strict logic sequencing at sub-second speeds.")
with t2:
    st.markdown("#### 🔍 Nomic-Embed-v1.5")
    st.caption("**Semantic Retrieval**")
    st.write("Eliminates manual jargon mapping by mathematically clustering synonyms like 'Client' and 'Customer'.")
with t3:
    st.markdown("#### 📑 Ontology & Schema")
    st.caption("**The Guardrails**")
    st.write("The JSON-LD enforces mandatory join logic and follows semantic knowledge of teh data, while the Markdown provides the technical source of truth.")


with st.sidebar:
    st.header("🌐 Past User queries")
    for h in history: 
        # Safely handle older queries that might not have a location tracked yet
        loc = h.get('user_location') or "Unknown Location"
        st.info(f"📍 {loc}\n🗨️ {h['query']}")

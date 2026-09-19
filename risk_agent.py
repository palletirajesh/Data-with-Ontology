import json
import re
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
from sentence_transformers import SentenceTransformer

from semantic_gateway import AccessDenied, GatewayError, SemanticGateway

st.set_page_config(page_title="Risk Data Agent v2", page_icon="🏦", layout="wide")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
TOGETHER_API_KEY = st.secrets.get("TOGETHER_API_KEY1", "")
BQ_PROJECT = st.secrets["bigquery"]["project_id"]
BQ_DATASET = st.secrets["bigquery"]["dataset_id"]
APP_ROLE = st.secrets.get("APP_ROLE", "RiskAnalyst")
MAX_BYTES_BILLED = int(st.secrets.get("MAX_BYTES_BILLED", 1_000_000_000))
HISTORY_TABLE = f"{BQ_PROJECT}.{BQ_DATASET}.query_history"


def get_bq_client():
    info = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(info)
    return bigquery.Client(credentials=creds, project=info["project_id"])


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_gateway(project, dataset, role):
    return SemanticGateway(
        "knowledge_base.jsonld",
        "application_policy.json",
        load_embedder(),
        project,
        dataset,
        role,
    )


bq_client = get_bq_client()
gateway = load_gateway(BQ_PROJECT, BQ_DATASET, APP_ROLE)


def call_llm(prompt):
    """The external LLM receives user language only, never database/ontology metadata."""
    providers = [
        (
            "https://api.groq.com/openai/v1/chat/completions",
            GROQ_API_KEY,
            "llama-3.3-70b-versatile",
            12,
        )
    ]
    if TOGETHER_API_KEY:
        providers.append(
            (
                "https://api.together.xyz/v1/chat/completions",
                TOGETHER_API_KEY,
                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                15,
            )
        )
    for url, key, model, timeout in providers:
        try:
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
                timeout=timeout,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except requests.RequestException:
            pass
    raise GatewayError("No language-model provider returned a valid intent.")


def intent_from_prompt(user_text):
    prompt = f"""
You are only a language parser. You do not know the database schema, table names,
column names, joins, ontology, capability keys, credentials, project, or dataset.
Never generate SQL and never invent database metadata.

Return ONLY a JSON object:
{{
  "operation": "retrieve" or "count",
  "requested_attributes": ["ordinary business phrase"],
  "filters": [{{"field":"ordinary business phrase","operator":"eq|ne|gt|gte|lt|lte|in|contains|between","value":"literal/number/boolean/date/array"}}]
}}

Rules:
- Use phrases from the request or plain-language equivalents.
- Do not return physical database identifiers or SQL fragments.
- Do not invent numeric thresholds.
- "how many" means operation=count.
- late/overdue payments without a number means payment delay > 0.
- a brand such as Amazon is a filter value; describe its field generically as retail partner.
- keep requested_attributes minimal; filter fields need not be repeated there.

User request: {json.dumps(user_text)}
"""
    raw = call_llm(prompt).replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise GatewayError("The language model returned invalid logical intent.")
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise GatewayError("Logical intent must be a JSON object.")
    return obj


def query_parameters(specs):
    out = []
    for p in specs:
        if p.get("array"):
            out.append(bigquery.ArrayQueryParameter(p["name"], p["type"], p["value"]))
        else:
            out.append(bigquery.ScalarQueryParameter(p["name"], p["type"], p["value"]))
    return out


def run_compiled(compiled):
    params = query_parameters(compiled["parameters"])
    dry_cfg = bigquery.QueryJobConfig(
        dry_run=True,
        use_query_cache=False,
        query_parameters=params,
        maximum_bytes_billed=MAX_BYTES_BILLED,
    )
    dry = bq_client.query(compiled["sql"], job_config=dry_cfg)
    exec_cfg = bigquery.QueryJobConfig(
        query_parameters=params,
        maximum_bytes_billed=MAX_BYTES_BILLED,
    )
    df = bq_client.query(compiled["sql"], job_config=exec_cfg).result(timeout=30).to_dataframe(
        create_bqstorage_client=False
    )
    return df, int(dry.total_bytes_processed or 0)


def load_history():
    try:
        return bq_client.query(
            f"SELECT user_query, generated_sql FROM `{HISTORY_TABLE}` ORDER BY timestamp DESC LIMIT 25"
        ).to_dataframe()
    except Exception:
        return pd.DataFrame(columns=["user_query", "generated_sql"])


def save_history(user_text, sql):
    try:
        bq_client.insert_rows_json(
            HISTORY_TABLE,
            [{"user_query": user_text, "generated_sql": sql, "timestamp": datetime.now().isoformat()}],
        )
    except Exception:
        pass


if "history" not in st.session_state:
    st.session_state.history = load_history()

with st.sidebar:
    st.header("Query history")
    st.caption("History is informational; executable SQL is never reused.")
    if st.button("Sync history", width="stretch"):
        st.session_state.history = load_history()
    for i, row in st.session_state.history.iterrows():
        if st.button(row["user_query"], key=f"h{i}", width="stretch"):
            st.session_state.main_input = row["user_query"]
            st.rerun()
    st.markdown("---")
    st.caption(f"Application role: **{APP_ROLE}**")

st.title("🏦 Risk Data Agent v2")
st.caption(
    "External LLM → logical intent only. Ontology, capability keys, authorization, joins and SQL stay inside the application boundary."
)

try:
    with open("additional_data.xlsx", "rb") as f:
        st.download_button("Download project data", f, "additional_data.xlsx")
except FileNotFoundError:
    pass

user_input = st.text_input(
    "What risk data do you need?",
    key="main_input",
    placeholder="e.g. Amazon customers more than 30 days overdue...",
)

if st.button("Build & execute authorized query", type="primary", disabled=not bool(user_input)):
    try:
        with st.spinner("Creating logical intent without exposing metadata..."):
            intent = intent_from_prompt(user_input)
        with st.spinner("Resolving ontology capabilities and policy..."):
            compiled = gateway.compile(intent)
        with st.spinner("Dry-running and executing approved SQL..."):
            df, estimated = run_compiled(compiled)
        st.session_state.last_intent = intent
        st.session_state.last_compiled = compiled
        st.session_state.last_df = df
        st.session_state.last_estimated = estimated
        save_history(user_input, compiled["sql"])
        st.session_state.history = load_history()
    except AccessDenied as e:
        st.error(f"Access denied: {e}")
    except GatewayError as e:
        st.error(f"Query blocked: {e}")
    except Exception as e:
        st.error(f"Query execution failed: {e}")

if "last_compiled" in st.session_state:
    left, right = st.columns([1, 1.5])
    with left:
        st.subheader("Application decision")
        st.caption("Logical intent from external LLM")
        st.json(st.session_state.last_intent)
        st.caption("Internally authorized capabilities")
        for cap in st.session_state.last_compiled["capabilities"]:
            st.code(f"{cap['key']} | {cap['column']} | {cap['classification']}", language=None)
        st.metric("Dry-run bytes", f"{st.session_state.last_estimated:,}")
    with right:
        st.subheader("Deterministic SQL")
        st.caption("Read-only. Users and external LLMs cannot edit executable SQL.")
        st.code(st.session_state.last_compiled["sql"], language="sql")
        if st.session_state.last_compiled["parameters"]:
            st.caption("Bound parameters")
            st.json(st.session_state.last_compiled["parameters"])
        st.subheader("Results")
        st.dataframe(st.session_state.last_df, use_container_width=True)

st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
for col, title, desc in [
    (c1, "External LLM", "Language → intent"),
    (c2, "Ontology", "Local semantic resolution"),
    (c3, "Capability Gate", "Role + operator policy"),
    (c4, "SQL Compiler", "Deterministic + parameterized"),
    (c5, "BigQuery", "Dry-run + bounded execution"),
]:
    with col:
        st.markdown(f"**{title}**")
        st.caption(desc)

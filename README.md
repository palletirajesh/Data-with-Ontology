# Risk Data Agent v2 — LLM + Ontology + Capability Gateway

This project converts natural-language risk questions into authorized, parameterized BigQuery queries while keeping physical database metadata out of the external LLM context.

## Architecture

```text
User prompt
  → External LLM (logical JSON intent only)
  → Semantic Gateway (local JSON-LD resolution)
  → Capability Gate (role + operator policy)
  → Deterministic SQL compiler
  → BigQuery dry run
  → BigQuery execution
```

The external LLM never receives JSON-LD, table names, column names, joins, project/dataset names, capability keys, or credentials.

## Application-layer keys

`application_policy.json` is the application-side binding layer. Each approved ontology column gets a deterministic internal key:

```text
CAP_<first 12 hex characters of SHA256(ontology-column-id)>
```

The key is an internal identifier, not a password. Execution also requires the application role, an allowed operator, an ontology-approved join path, SQL policy validation, and a BigQuery dry run.

Example LLM output:

```json
{
  "operation": "retrieve",
  "requested_attributes": ["credit score"],
  "filters": [
    {"field": "retail partner", "operator": "eq", "value": "Amazon"},
    {"field": "late payments", "operator": "gt", "value": 30}
  ]
}
```

The application resolves those phrases locally and creates the SQL. Users and external LLMs cannot edit executable SQL.

## Safety changes from v1

- No schema/JSON-LD in the external LLM prompt.
- No LLM-generated executable SQL.
- No arbitrary SQL editor.
- No reuse of complete SQL strings from semantic history.
- Query values use BigQuery parameters.
- Only `SELECT` is allowed; DML/DDL and `SELECT *` are rejected.
- Referenced tables must be in the ontology-approved join plan.
- BigQuery dry-run and `maximum_bytes_billed` are applied before execution.
- Query history is saved only after successful execution.
- `ssn_hash` is configured as `RESTRICTED` for `RiskManager`/`Admin`, not `RiskAnalyst`.

## Files

- `risk_agent.py` — Streamlit UI and external intent boundary.
- `semantic_gateway.py` — local resolver, capability gate, join planner and SQL compiler.
- `application_policy.json` — application-side physical/access policy bound to JSON-LD IDs.
- `knowledge_base.jsonld` — ontology and approved relationships.
- `database_schema.md` — documentation only; no longer sent to the LLM.

## Secrets

```toml
GROQ_API_KEY = "..."
TOGETHER_API_KEY1 = "..." # optional
APP_ROLE = "RiskAnalyst"
MAX_BYTES_BILLED = 1000000000

[bigquery]
project_id = "..."
dataset_id = "..."

[gcp_service_account]
# standard service account fields
```

Run with `streamlit run risk_agent.py`.

## Next step

Reintroduce ontology learning as a governed proposal flow: correction → candidate assertion → validation → Git branch/PR → human approval → production ontology. Do not let an LLM write directly to `main` or change production capabilities automatically.

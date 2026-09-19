import hashlib
import json
import re
from collections import deque


class GatewayError(ValueError):
    pass


def _list(v):
    return v if isinstance(v, list) else ([] if v is None else [v])


def _norm(v):
    return re.sub(r"[^a-z0-9]+", " ", str(v).lower()).strip()


def _short(v):
    return str(v).split(":", 1)[-1]


def capability_key(ontology_id):
    return "CAP_" + hashlib.sha256(ontology_id.encode()).hexdigest()[:12].upper()


class SemanticGateway:
    """Trusted local layer: ontology -> capability -> policy -> deterministic SQL."""

    def __init__(self, ontology_path, policy_path, embedder, project, dataset):
        with open(ontology_path, encoding="utf-8") as f:
            self.kb = json.load(f)
        with open(policy_path, encoding="utf-8") as f:
            self.policy = json.load(f)
        self.embedder = embedder
        self.project, self.dataset = project, dataset
        self.nodes = {n.get("@id"): n for n in self.kb.get("@graph", []) if n.get("@id")}
        self.root = self.policy["root_table"]
        self.col_table, self.joins = {}, {}
        self._index_graph()
        self.caps, self.aliases, self.alias_cap = {}, [], []
        self._index_capabilities()
        self.alias_vectors = self.embedder.encode(self.aliases, normalize_embeddings=True)

    def _index_graph(self):
        for tid, node in self.nodes.items():
            if node.get("@type") != "bank:Table":
                continue
            for cid in _list(node.get("hasColumn")):
                self.col_table[cid] = tid
            for j in _list(node.get("joinsWith")):
                target = j.get("targetTable")
                if not target:
                    continue
                fwd = (tid, target, j["sourceKey"], j["targetKey"])
                rev = (target, tid, j["targetKey"], j["sourceKey"])
                self.joins.setdefault(tid, []).append(fwd)
                self.joins.setdefault(target, []).append(rev)

    def _ontology_aliases(self, ontology_id):
        out = []
        node = self.nodes.get(ontology_id, {})
        if node.get("rdfs:label"):
            out.append(node["rdfs:label"])
        concept_ids = _list(node.get("representsConcept")) + _list(node.get("bank:representsConcept"))
        for cid in concept_ids:
            concept = self.nodes.get(cid, {})
            out += _list(concept.get("businessJargon")) + _list(concept.get("bank:businessJargon"))
            if concept.get("rdfs:label"):
                out.append(concept["rdfs:label"])
        return out

    def _index_capabilities(self):
        referenced_ids = set(self.col_table)
        for spec in self.policy["capabilities"]:
            oid = spec["ontology_id"]
            if oid not in self.nodes and oid not in referenced_ids:
                raise GatewayError(f"Policy references ontology id not present in JSON-LD: {oid}")
            table = self.col_table.get(oid) or spec.get("table")
            if not table:
                raise GatewayError(f"No source table for {oid}")
            node = self.nodes.get(oid, {})
            column = spec.get("column") or node.get("rdfs:label")
            if not column:
                raise GatewayError(f"No physical column label for {oid}")
            key = capability_key(oid)
            cap = dict(spec, key=key, table=table, column=column)
            self.caps[key] = cap
            terms = [column, _short(oid).replace("_", " ")] + spec.get("aliases", []) + self._ontology_aliases(oid)
            for term in terms:
                if _norm(term):
                    self.aliases.append(str(term))
                    self.alias_cap.append(key)

    def resolve(self, phrase):
        p = _norm(phrase)
        exact = [self.alias_cap[i] for i, a in enumerate(self.aliases) if _norm(a) == p]
        if exact:
            return self._prefer(exact)
        q = self.embedder.encode([phrase], normalize_embeddings=True)[0]
        scores = self.alias_vectors @ q
        idx = int(scores.argmax())
        if float(scores[idx]) < float(self.policy.get("semantic_threshold", 0.62)):
            raise GatewayError(f"No approved ontology concept confidently matches '{phrase}'.")
        return self.caps[self.alias_cap[idx]]

    def _prefer(self, keys):
        root = [k for k in keys if self.caps[k]["table"] == self.root]
        return self.caps[(root or keys)[0]]

    def _validate_capability(self, cap, operator=None):
        if operator and operator not in cap["operators"]:
            raise GatewayError(
                f"Operator {operator} is not approved for capability {cap['key']}."
            )

    def _path(self, target):
        if target == self.root:
            return []
        q = deque([(self.root, [])]); seen = {self.root}
        while q:
            table, path = q.popleft()
            for edge in self.joins.get(table, []):
                nxt = edge[1]
                if nxt in seen:
                    continue
                np = path + [edge]
                if nxt == target:
                    return np
                seen.add(nxt); q.append((nxt, np))
        raise GatewayError(f"No ontology-approved join path to {_short(target)}.")

    def compile(self, intent):
        operation = intent.get("operation", "retrieve")
        if operation not in {"retrieve", "count"}:
            raise GatewayError("Only retrieve and count operations are supported.")

        selected = [c for c in self.caps.values() if c.get("default")]
        generic = {"customer", "customers", "record", "records", "details", "data"}
        for attr in intent.get("requested_attributes", []):
            if _norm(attr) in generic:
                continue
            cap = self.resolve(attr); self._validate_capability(cap)
            if cap["key"] not in {x["key"] for x in selected}:
                selected.append(cap)

        filters = []
        for f in intent.get("filters", []):
            cap = self.resolve(f.get("field", "")); op = f.get("operator", "eq")
            self._validate_capability(cap, op)
            if "value" not in f:
                raise GatewayError(f"Missing value for {f.get('field')}.")
            filters.append((cap, op, f["value"]))
            if cap["key"] not in {x["key"] for x in selected}:
                selected.append(cap)

        for cap in selected:
            self._validate_capability(cap)
        tables = {c["table"] for c in selected}
        edges = []
        for t in tables:
            for e in self._path(t):
                if e not in edges:
                    edges.append(e)

        aliases = {self.root: "c"}; joined = {self.root}; joins = []; counter = 1
        pending = list(edges)
        while pending:
            progressed = False
            for e in list(pending):
                src, dst, sk, dk = e
                if src not in joined:
                    continue
                if dst not in joined:
                    aliases[dst] = f"t{counter}"; counter += 1
                    joins.append(f"JOIN `{self.project}.{self.dataset}.{_short(dst)}` {aliases[dst]} ON {aliases[src]}.{sk} = {aliases[dst]}.{dk}")
                    joined.add(dst)
                pending.remove(e); progressed = True
            if not progressed:
                raise GatewayError("Could not build approved join plan.")

        if operation == "count":
            idcap = next(c for c in selected if c.get("default") and c["column"] == "cust_id")
            select = f"SELECT COUNT(DISTINCT {aliases[idcap['table']]}.cust_id) AS record_count"
        else:
            refs, seen = [], set()
            for c in selected:
                ref = (c["table"], c["column"])
                if ref not in seen:
                    seen.add(ref); refs.append(f"{aliases[c['table']]}.{c['column']}")
            select = "SELECT\n    " + ",\n    ".join(refs)

        params, where = [], []
        for i, (cap, op, value) in enumerate(filters):
            name = f"p{i}"; ref = f"{aliases[cap['table']]}.{cap['column']}"; typ = cap["type"]
            if op == "between":
                if not isinstance(value, list) or len(value) != 2:
                    raise GatewayError("between requires two values")
                a, b = name + "a", name + "b"
                params += [{"name": a, "type": typ, "value": value[0]}, {"name": b, "type": typ, "value": value[1]}]
                where.append(f"{ref} BETWEEN @{a} AND @{b}")
            elif op == "in":
                vals = value if isinstance(value, list) else [value]
                params.append({"name": name, "type": typ, "value": vals, "array": True})
                where.append(f"{ref} IN UNNEST(@{name})")
            else:
                params.append({"name": name, "type": typ, "value": value})
                if op == "contains": where.append(f"CONTAINS_SUBSTR({ref}, @{name})")
                elif typ == "STRING" and op in {"eq", "ne"}: where.append(f"UPPER({ref}) {'=' if op == 'eq' else '!='} UPPER(@{name})")
                else:
                    symbol = {"eq":"=","ne":"!=","gt":">","gte":">=","lt":"<","lte":"<="}.get(op)
                    if not symbol: raise GatewayError(f"Unsupported operator {op}")
                    where.append(f"{ref} {symbol} @{name}")

        sql = "\n".join([select, f"FROM `{self.project}.{self.dataset}.{_short(self.root)}` c"] + joins + (["WHERE\n    " + "\n    AND ".join(where)] if where else []))
        self.validate(sql, {f"{self.project}.{self.dataset}.{_short(t)}" for t in joined})
        return {"sql": sql, "parameters": params, "capabilities": [{"key":c["key"],"column":c["column"],"classification":c["classification"]} for c in selected]}

    @staticmethod
    def validate(sql, allowed_tables):
        if not sql.lstrip().upper().startswith("SELECT") or ";" in sql or re.search(r"SELECT\s+\*", sql, re.I):
            raise GatewayError("SQL policy rejected the compiled query.")
        if re.search(r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE|CALL|EXPORT|LOAD|GRANT|REVOKE)\b", sql, re.I):
            raise GatewayError("Prohibited SQL operation.")
        refs = set(re.findall(r"`([^`]+)`", sql))
        if refs - allowed_tables:
            raise GatewayError("Query referenced a table outside the approved ontology plan.")

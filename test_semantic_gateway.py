import unittest

from semantic_gateway import AccessDenied, GatewayError, SemanticGateway


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        # Exact aliases are used in these policy tests; vectors only satisfy gateway initialization.
        return [[1.0] for _ in texts]


class SemanticGatewayTests(unittest.TestCase):
    def gateway(self, role="RiskAnalyst"):
        return SemanticGateway(
            "knowledge_base.jsonld",
            "application_policy.json",
            FakeEmbedder(),
            "test-project",
            "test_dataset",
            role,
        )

    def test_compiles_parameterized_authorized_query(self):
        compiled = self.gateway().compile(
            {
                "operation": "retrieve",
                "requested_attributes": ["credit score"],
                "filters": [
                    {"field": "retail partner", "operator": "eq", "value": "Amazon"},
                    {"field": "late payments", "operator": "gt", "value": 30},
                ],
            }
        )
        self.assertIn("@p0", compiled["sql"])
        self.assertIn("@p1", compiled["sql"])
        self.assertNotIn("Amazon", compiled["sql"])
        self.assertIn("dim_card_association", compiled["sql"])
        self.assertIn("fact_card_ledger", compiled["sql"])
        self.assertIn("fact_credit_bureau", compiled["sql"])

    def test_restricted_identity_is_denied_to_risk_analyst(self):
        with self.assertRaises(AccessDenied):
            self.gateway().compile(
                {"operation": "retrieve", "requested_attributes": ["ssn"], "filters": []}
            )

    def test_sql_policy_rejects_mutation(self):
        with self.assertRaises(GatewayError):
            SemanticGateway.validate(
                "DELETE FROM `test-project.test_dataset.dim_customer` WHERE 1=1",
                {"test-project.test_dataset.dim_customer"},
            )


if __name__ == "__main__":
    unittest.main()

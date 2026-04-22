# Bank Database Schema & Business Glossary
You MUST use fully qualified Databricks table names (catalog.schema.table). Always use the standard table aliases provided below when writing JOINs.

---

### TABLE: fact_card_ledger (Alias: l)
**Columns:**
- `card_id` (string): Card id for the customer
- `used_amount` (decimal): The amount consumed on the credit card
- `unutilized_amount` (decimal): The amount not used from the credit card limit
- `committed_limit` (decimal): Maximum amount the customer can use
- `payment_due_amount` (decimal): Amount due for payment as of date
- `actual_payment_made` (decimal): Amount paid back to the bank towards the due
- `days_past_due` (int): Number of days the credit card payments are not made by the customer
- `as_of_date` (date): Date on which the payment is due

**Business Jargon & Semantic Rules:**
- **ENR** or **Balance** = `used_amount`
- **OSUC** or **Available Credit** = `unutilized_amount`
- **Card Number** / **Account Number** = `card_id`
- **Late payments** / **Payment delay** / **Overdue** = `days_past_due`
- Also covers concepts: debt, usage, ledger, unutilized, limit.

---

### TABLE: dim_customer (Alias: c)
**Columns:**
- `cust_id` (string): Unique bank identifier
- `customer_name` (string): Full name of the customer
- `ssn_hash` (string): SIN number / Social Insurance Number
- `card_id` (string): Unique card ID
- `internal_banking_flag` (boolean): Indicates if they have a deposit account

**Business Jargon & Semantic Rules:**
- **Card Number** / **Account Number** = `card_id`
- **Customer ID** = `cust_id`
- Also covers concepts: client list, identity, ssn, SIN.

---

### TABLE: fact_credit_bureau (Alias: f)
**Columns:**
- `ssn_hash` (string): SIN number of the credit card customer
- `fico_score` (int): Credit score or FICO score
- `refresh_date` (date): Date when this score was refreshed

**Business Jargon & Semantic Rules:**
- Also covers concepts: FICO, credit score, bureau refresh.

---

### TABLE: dim_card_association (Alias: a)
**Columns:**
- `card_id` (string): Unique card ID
- `card_partner` (string): Partner names (e.g., Amazon, Target)
- `geography_group` (string): Bank region

**Business Jargon & Semantic Rules:**
- **Retail Partner** = `card_partner`
- Also covers concepts: Amazon card, Target card, partner, retail.

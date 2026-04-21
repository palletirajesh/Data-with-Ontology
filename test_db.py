from databricks import sql
import os

try:
    conn = sql.connect(
        server_hostname="YOUR_HOSTNAME", # No https://
        http_path="YOUR_HTTP_PATH",
        access_token="YOUR_TOKEN"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    print("✅ Connection Successful!")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ Connection Failed: {e}")

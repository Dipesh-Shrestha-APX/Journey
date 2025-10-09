from fastmcp import FastMCP
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import json

# --- Configuration ---
DB_NAME = os.getenv("POSTGRES_DB", "expenses_db")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "dipesh@123")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "categories.json")

mcp = FastMCP("ExpenseTracker")

# --- Helper function for connections ---
def get_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )

# --- Initialize database ---
def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS expenses (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    amount NUMERIC(10,2) NOT NULL,
                    category VARCHAR(100) NOT NULL,
                    subcategory VARCHAR(100) DEFAULT '',
                    note TEXT DEFAULT ''
                )
            """)
            conn.commit()

init_db()

# --- Add Expense ---
@mcp.tool()
def add_expense(date, amount, category, subcategory="", note=""):
    '''Add a new expense entry to the PostgreSQL database.'''
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO expenses (date, amount, category, subcategory, note)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (date, amount, category, subcategory, note)
            )
            new_id = cur.fetchone()[0]
            conn.commit()
            return {"status": "ok", "id": new_id}

# --- List Expenses ---
@mcp.tool()
def list_expenses(start_date, end_date):
    '''List expense entries within an inclusive date range.'''
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, date, amount, category, subcategory, note
                FROM expenses
                WHERE date BETWEEN %s AND %s
                ORDER BY id ASC
                """,
                (start_date, end_date)
            )
            return cur.fetchall()

# --- Summarize Expenses ---
@mcp.tool()
def summarize(start_date, end_date, category=None):
    '''Summarize expenses by category within an inclusive date range.'''
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT category, SUM(amount) AS total_amount
                FROM expenses
                WHERE date BETWEEN %s AND %s
            """
            params = [start_date, end_date]

            if category:
                query += " AND category = %s"
                params.append(category)

            query += " GROUP BY category ORDER BY category ASC"

            cur.execute(query, params)
            return cur.fetchall()

# --- Delete the Expense Record ---
@mcp.tool()
def delete_expense(date):
    """
    Delete expense entries for a specific date.
    Returns the number of rows deleted.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            # --- Correct table name and query ---
            query = "DELETE FROM expenses WHERE date = %s"
            
            # Execute with parameter tuple
            cur.execute(query, (date,))
            
            # Commit changes
            conn.commit()
            
            # Return result
            if cur.rowcount == 0:
                return f"No records found for {date}"
            return f"{cur.rowcount} record(s) deleted for {date}"

# Update the expenses table at particular date with new data 
@mcp.tool()
def update_expense(date, amount, category, subcategory="", note=""):
    """Update expense details for a specific date."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            query = """
                UPDATE expenses
                SET amount = %s,
                    category = %s,
                    subcategory = %s,
                    note = %s
                WHERE date = %s
            """
            params = (amount, category, subcategory, note, date)
            cur.execute(query, params)
            conn.commit()
            return f"{cur.rowcount} record(s) updated for {date}"

# --- Resource: Categories JSON ---
@mcp.resource("expense://categories", mime_type="application/json")
def categories():
    # Read fresh each time so you can edit the file without restarting
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    mcp.run()

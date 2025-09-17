import psycopg2
from psycopg2 import sql

# Define connection
conn = psycopg2.connect(host="localhost", database="dipesh", user="postgres", password="dipesh@123", port=5432)

# Define cursor
cur = conn.cursor()

# Autocommit connection
conn.autocommit = True

# SQL definition and execution

## Database creation
# db_name = "dipesh"
# cur.execute(sql.SQL("CREATE DATABASE {db}}").format(db = sql.Identifier(db_name)))

## Table creation
tb_name = "Person"
sqlCode = sql.SQL("""
                  CREATE TABLE IF NOT EXISTS {tb}(
                    id INT PRIMARY KEY,
                    name VARCHAR(255),
                    gender CHAR
                  );
                  """)

cur.execute(sqlCode.format(tb = sql.Identifier(tb_name)))

cur.execute(sql.SQL("""
    INSERT INTO Person(id, name, gender) VALUES
                    (1, "ram", M),
                    (1, "ram", M),
                    (1, "ram", M),
                    """))

# Close cursor
cur.close()
# Close connection
conn.close()

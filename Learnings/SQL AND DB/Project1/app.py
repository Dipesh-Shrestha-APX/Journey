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

## Table deletion or dropping
# tb_name = "Person"
# sqlCode = sql.SQL("DROP TABLE {tb};")
# cur.execute(sqlCode.format(tb = sql.Identifier(tb_name)))
            
## Table creation
tb_name = "person"
# tb_name = "Person" 
  ## This way makes case sensitive
  ## And you would need to call like (select * from "Person")
sqlCode = sql.SQL("""
                  CREATE TABLE IF NOT EXISTS {tb}(
                    id INT PRIMARY KEY,
                    name VARCHAR(255),
                    gender CHAR
                  );
                  """)

cur.execute(sqlCode.format(tb = sql.Identifier(tb_name)))

insertQuery = sql.SQL("""
    INSERT INTO {tb}(id, name, gender) VALUES
                    (4, 'hari', 'M'),
                    (6, 'radha', 'F'),
                    (5, 'laxmi', 'F') ;
                    """)
cur.execute(insertQuery.format(tb = sql.Identifier(tb_name)))

# Close cursor
cur.close()
# Close connection
conn.close()

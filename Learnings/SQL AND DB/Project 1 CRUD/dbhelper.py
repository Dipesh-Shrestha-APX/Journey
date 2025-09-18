# !pip install psycopg2-binary
import psycopg2 
import sys
from psycopg2 import sql
class DBConn:
  def __init__(self):
    try:
      self.dbconn = psycopg2.connect(host="localhost", database="dipesh", user="postgres", password="dipesh@123", port=5432)
      self.cur = self.dbconn.cursor()
    except:
      print("The DB connection failed")
      sys.exit(0)
    else:
      print("Connection has been established")
  
  def register(self):
    name = input("Enter your name: ")
    gender = input("Enter your gender: ")
    try:
      self.dbconn.autocommit = True
      query = sql.SQL("INSERT INTO person(name, gender) VALUES(%s,%s);""")
      print(query)
      self.cur.execute(query, (name,gender))
      # query = sql.SQL("""
      #                 INSERT INTO person (name, gender) VALUES ({name}, {gender});""").format(
      #                 name=sql.Literal(name),
      #                 gender=sql.Literal(gender)
      #                 )
      # self.cur.execute(query)
    except:
      print("New user can't be registered")
    else:
      print("Successfully registered")
  
  #Validate Function
  def validate(self):
    name = input("Enter your name: ")
    gender = input("Enter your gender: ")
    selectQuery = sql.SQL("""SELECT * FROM person WHERE name=%s AND gender=%s""")
    self.cur.execute(selectQuery,(name, gender))
    list = self.cur.fetchall()
    # print(list)
    if len(list)==0:
      print("You're not registered here")
    else:
      print("Yes you're registered here")
  
  #Update Function
  def update(self):
    name = input("Enter your name: ")
    gender = input("Enter your gender: ")
    new_name = input("Enter your new name: ")
    new_gender = input("Enter your new gender: ")
    selectQuery = sql.SQL("""
                          UPDATE person SET name=%s, gender=%s WHERE name=%s AND gender=%s""")
    try:
      self.cur.execute(selectQuery,(new_name, new_gender, name, gender))
    except:
      print("Some error occurred couldn't be updated")
    else:
      print("Updated the table")
    
  
  #Delete Function
  def delete(self):
    name = input("Enter your name: ")
    gender = input("Enter your gender: ")
    selectQuery = sql.SQL("""DELETE FROM person WHERE name=%s AND gender=%s""")
    try:
      self.cur.execute(selectQuery,(name, gender))
    except:
      print("Record couldn't be deleted")
    else:
      print("Record deleted successfully")
    
  
  
from dbhelper import DBConn
class CrudDB:
  def __init__(self):
    self.dbConn = DBConn()
    self.menu()
  def menu(self):
    option = input("""
          Select the option:
          1) Register new user
          3) Validate the user
          3) Update the user info
          4) Delete the user
    """)
    
    # print("The option you have chosen is {}".format(option))

    if int(option)==1:
      self.dbConn.register()
      self.menu()

    elif int(option)==2:
      self.dbConn.validate()
      self.menu()

    elif int(option)==3:
      self.dbConn.update()
      self.menu()

    elif int(option)==4:
      self.dbConn.delete()
      self.menu()

    else:
      print("Choose valid option please")
      self.menu()


crudObj = CrudDB()

    
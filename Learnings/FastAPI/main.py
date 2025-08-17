from fastapi import FastAPI

app = FastAPI()

#Defining a route to your endpoint

#the request that will come to this endpoint is get request i.e getting some data from API
# @app is called decorator
# @app.get("Route or the URL is defined here")
  #e.g. www.google.com/ will hit this endpoint

#We have created a home route / that listens to get request on the /
@app.get("/")
def hello():
  return {'message': "Hello This is GOD"}

# Creating another endpoint
@app.get('/about')
def about():
  return {'message': 'This is my FastAPI for this recommendation prohject'}
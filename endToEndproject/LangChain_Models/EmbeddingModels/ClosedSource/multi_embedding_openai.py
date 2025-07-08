from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents=[
  "I am god himself",
  "Kathmandu is best place on earth"  
]

result = embedding.embed_documents(documents)

print(str(result))
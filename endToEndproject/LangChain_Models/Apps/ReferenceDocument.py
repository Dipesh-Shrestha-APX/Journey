from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
  "Paris is the capital of France.",
  "Kathmandu is the capital of Nepal.",
  "Tokyo is the capital of Japan.",
  "Brasília is the capital of Brazil.",
  "Canberra is the capital of Australia."
]


query = "Info about capital of Brazil"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

#Simlarity between the doc and the query text and in 1D list
scores = cosine_similarity([query_embedding],doc_embeddings)[0]

index,score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(documents[index])
print(f"Similarity score of {score}")
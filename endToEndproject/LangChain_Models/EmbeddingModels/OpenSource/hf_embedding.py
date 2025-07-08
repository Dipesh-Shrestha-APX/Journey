from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text = "I love my family"

vector = embedding.embed_query(text)

print(str(vector))

#Mutliple documents embedding generation
documents=[
  "I am god himself",
  "Kathmandu is best place on earth"  
]

vector2 = embedding.embed_documents(documents)
print(vector2)

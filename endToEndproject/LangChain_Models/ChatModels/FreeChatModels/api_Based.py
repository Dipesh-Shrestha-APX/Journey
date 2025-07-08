from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id="deepseek-ai/DeepSeek-R1-0528",
  task="text-generation"
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("Introduce yourself in nepali in 1 line")

print(result.content)
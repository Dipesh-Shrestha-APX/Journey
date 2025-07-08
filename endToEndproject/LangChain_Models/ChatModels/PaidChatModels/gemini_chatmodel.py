from langchain_openai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=1.5, max_completion_tokens='')
  ## temperature is the parameter that controls the randomness of language model's output.
  #  it affects how creative or deterministic the response are 
  # lower values 0.0-0.3 More deterministic
  # higher values 0.7-1.5 More random and creative and diverse

  ## max_completion_tokens will allow you to control the no. of tokens 
  # and will help control the pricing on paid models

result = model.invoke("Suggest me unique company names that is inspired by gods and divine realm")

print(result.content)
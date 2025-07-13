from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

import re
import streamlit as st

load_dotenv()


llm = HuggingFaceEndpoint(
  repo_id="deepseek-ai/DeepSeek-R1-0528",
  task="text-generation"
)
model = ChatHuggingFace(llm=llm)

############# Streamlit app section #####################
st.header("Streamlit based App for Convo")

user_text = st.text_input("Enter your prompt")

st.subheader("Want to test the dynamic prompt ?")

country_input = st.selectbox("Select the nation", ["Nepal", "China", "India", "USA", "Germany"])

info_on = st.selectbox("What do you want info on", ["capital", "area", "population"])

description_depth = st.selectbox("Description level", ["one word", "one line", "one paragraph"])

### prompt template
from langchain_core.prompts import PromptTemplate

my_template = PromptTemplate(
    input_variables=["country_input_temp", "info_on_temp", "description_depth_temp"],
    template="""
<human>You are an intelligent assistant. Provide factual and clear information.

Task: Give information about {info_on_temp} of {country_input_temp} in {description_depth_temp}.

Answer format:
<think>Explain how you reach the answer with internal reasoning.</think> Final output here.</human>
"""
)

# Correct way to fill in the template
my_dynamic_prompt = my_template.format_prompt(
    country_input_temp=country_input,
    info_on_temp=info_on,
    description_depth_temp=description_depth
).to_string()

if st.button("Ask DeepSeek"):
    prompt_to_invoke = user_text.strip() if user_text.strip() else my_dynamic_prompt
    if not user_text.strip():
        st.warning("No user input provided. Using dynamic prompt based on selections.")

    try:
        result = model.invoke(prompt_to_invoke)
        final_answer = re.sub(r"<think>.*?</think>", "", result.content, flags=re.DOTALL).strip()
    except Exception as e:
        st.error(f"Error processing model output: {e}")
        final_answer = result.content.strip()

    st.write(final_answer)
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import streamlit as st
from langchain.llms import Cohere

# Set up Cohere API key
os.environ["COHERE_API_KEY"] = "Enter-your-api-key"

# Initialize Cohere LLM
llm = Cohere(model="command-r-plus", temperature=0.8)

# Streamlit UI
st.title("Welcome to Finapics")
input_text = st.text_input("Search the financial topic you want")

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=["name"],
    template="Give a brif description about {name}"
)
second_input_prompt= PromptTemplate(
    input_variables=['name'],
    template="How is {name} calculated?"
)
third_input_prompt = PromptTemplate(
    input_variables=["name"],
    template="Who introduced this {name}?"
)
fourth_input_prompt = PromptTemplate(
    input_variables=["Inventor"],
    template="Give the exact date on which {Inventor} introduced this?"
)

# Define Chains
chain1 = LLMChain(llm=llm, prompt=first_input_prompt, output_key="Definition")
chain2= LLMChain(llm=llm, prompt=second_input_prompt, output_key="Calculation")
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, output_key="Inventor")
chain4 = LLMChain(llm=llm, prompt=fourth_input_prompt, output_key="Date")

# Combine Chains into a SequentialChain
parent_chain = SequentialChain(
    chains=[chain1, chain2, chain3, chain4],
    input_variables=["name"],
    output_variables=["Definition", "Calculation", "Inventor", "Date"],
    verbose=True
)

# Process input and generate output
if input_text:
    output = parent_chain({"name": input_text})
    st.write("Definition:", output.get("Definition"))
    st.write("Calculation:", output.get("Calculation"))
    st.write("Introduced by:", output.get("Inventor"))
    st.write("Date:", output.get("Date"))

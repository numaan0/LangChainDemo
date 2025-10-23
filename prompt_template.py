from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
# This is about the prompt template in which we can reserve some space for dynamic input



# LLM Initialization
llm=ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0,
)

# Prompt Template Creation

template = PromptTemplate.from_template("List three skills needed {job} in {year} to be pro at it.")

prompt = template.invoke({"job":"Software Developer","year": "2025"})
print(prompt)

chain = template|llm

# LLM Invocation with Prompt Template
print(chain.invoke({"job":"Software Developer","year": "2025"}))
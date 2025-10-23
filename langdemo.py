from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

llm_cold=ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0,
)

llm_hot=ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=1
)

prompt= "Write a tagline for an ice cream shop."

print("Temprature 0 Output:")
print(llm_cold.invoke(prompt))

print("\nTemprature 1 Output:")
print(llm_hot.invoke(prompt))
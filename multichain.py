from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# This is about the prompt template in which we can reserve some space for dynamic input



# LLM Initialization
llm=ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0,
)

# Prompt Template Creation

extract_skills = PromptTemplate.from_template("Extract all the skills from this job posting and make them comma seperated:{job_posting}")


explain_skillls = PromptTemplate.from_template("Take these {skills} and explain each one in one sentence with bullet points.")
parser = StrOutputParser()
chain = (extract_skills|llm|parser|(lambda skills:{"skills":skills})|explain_skillls|llm|parser)



job_posting = """
Senior Backend Engineer
Requirements:
- 5+ years Python experience
- Microservices architecture
- Kubernetes deployment
- RESTful API design
"""



# LLM Invocation with Prompt Template
result = chain.invoke({"job_posting":job_posting})
print(result)
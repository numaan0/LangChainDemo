import traceback
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
# from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


try:
    job_posting = """
BACKEND DEVELOPER (JUNIOR)

Company: StartupXYZ
Location: Hybrid (NYC)
Experience: 1-2 years

REQUIRED SKILLS:
- Node.js or Python
- Basic understanding of REST APIs
- SQL databases (MySQL or PostgreSQL)
- Git and GitHub
- Basic Linux commands

NICE TO HAVE:
- Experience with Express.js or Flask
- Knowledge of MongoDB
- Understanding of authentication (JWT)
- Docker basics
- Agile/Scrum experience

WHAT YOU'LL DO:
- Build backend features for web applications
- Write unit tests for your code
- Debug and fix production issues
- Work closely with senior engineers

SALARY: $70k - $90k
"""


    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)


    @tool
    def extract_skills(job_description:str)->str:
        """Extract all the skills from the Job Description"""
        template = PromptTemplate.from_template("Extract all required nice to have skills from this job description and return in comma seperated way\n{job_description}")
        chain = (template | model | StrOutputParser())
        return chain.invoke({"job_description":job_description})
    
    @tool
    def generate_summary(skills:str)->str:
        """Generate a summary based on the skills"""
        return f"The candidate possesses the following skills: {skills}."

    @tool
    def search_resume(required_skills: str) -> str:
        """Search resume for skills matching the required skills list"""
        resume_skills = ["Python", "FastAPI", "PostgreSQL", "Docker"]
        return f"Found in resume: {', '.join(resume_skills)}"
    
    agent  = create_agent(
        model,
        tools=[generate_summary,extract_skills,search_resume],
        system_prompt="You are a job application assistant. Help candidates apply for jobs by extracting skills, and generating cover letters."
        )

    result = agent.invoke({
        "messages":[{"role": "user", "content": f"Help me apply for a Backend Engineer role. Extract skills from JD and add those skills only and generate a cover letter. This is the {job_posting}"}]
    })

    print("\n=== FINAL RESULT ===")
    print(result["messages"][-1].content)
    
except Exception:
    traceback.print_exc()
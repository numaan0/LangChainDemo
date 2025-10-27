import traceback
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
# from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
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
    def search_skills(query:str)-> str:
        """Search for the skills in the resume"""
        
        resume_skills = ["Python", "Microservices", "Kubernetes", "RESTful APIs", "Docker", "AWS", "CI/CD"]
        return f"Found skills: {', '.join(resume_skills)}"


    @tool
    def generate_summary(skills:str)->str:
        """Generate a summary based on the skills"""
        return f"The candidate possesses the following skills: {skills}."


    agent  = create_agent(
        model,
        tools=[search_skills, generate_summary],
        system_prompt="You are a job application assistant. Help candidates apply for jobs by extracting skills, and generating cover letters."
        )

    result = agent.invoke({
        "messages":[{"role": "user", "content": f"Help me apply for a Backend Engineer role. Extract skills and generate a cover letter. This is the {job_posting}"}]
    })

    print("\n=== FINAL RESULT ===")
    print(result["messages"][-1].content)
    
except Exception:
    traceback.print_exc()
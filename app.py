import os
import json
import warnings
from datetime import datetime
import streamlit as st

# Suppress runtime warnings
warnings.filterwarnings('ignore')

# CrewAI & LangChain Imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import GoogleSerperAPIWrapper

# Streamlit Page Configurations
st.set_page_config(
    page_title="Multi-Agent Job Search System",
    page_icon="🤖",
    layout="wide"
)

# ─── SIDEBAR CONFIGURATION (API KEYS) ──────────────────────────────────
st.sidebar.header("🔑 API Configurations")
st.sidebar.markdown(
    "Get your keys here: \n"
    "- [Groq Console](https://console.groq.com)\n"
    "- [Serper Dev](https://serper.dev)"
)

groq_key = st.sidebar.text_input("Groq API Key", type="password")
serper_key = st.sidebar.text_input("Serper API Key", type="password")

if not groq_key or not serper_key:
    st.info("👈 Please enter your Groq and Serper API keys in the sidebar to unlock the application.")
    st.stop()

# Set keys into environment variables dynamically
os.environ['GROQ_API_KEY'] = groq_key
os.environ['SERPER_API_KEY'] = serper_key

# ─── LLM SETUP ────────────────────────────────────────────────────────
llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    temperature=0.1,
    max_tokens=4096,
    groq_api_key=groq_key
)

creative_llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    temperature=0.6,
    max_tokens=4096,
    groq_api_key=groq_key
)

# ─── CUSTOM TOOLS DEFINITION ──────────────────────────────────

class JobSearchTool(BaseTool):
    name: str = 'job_search'
    description: str = 'Search for real job listings online. Input: job title + location.'
    def _run(self, query: str) -> str:
        try:
            search = GoogleSerperAPIWrapper(serper_api_key=os.environ['SERPER_API_KEY'], type='search', k=5)
            return search.run(f'{query} job opening site:linkedin.com OR site:indeed.com OR site:glassdoor.com')
        except Exception:
            try:
                return DuckDuckGoSearchRun().run(f'{query} job posting requirements')
            except Exception as e:
                return f'Search error: {e}'

class SalaryResearchTool(BaseTool):
    name: str = 'salary_research'
    description: str = 'Research salary ranges for a role/location.'
    def _run(self, query: str) -> str:
        try:
            search = GoogleSerperAPIWrapper(serper_api_key=os.environ['SERPER_API_KEY'], type='search', k=3)
            return search.run(f'{query} salary range glassdoor OR levels.fyi')
        except Exception:
            return DuckDuckGoSearchRun().run(f'{query} average salary')

class CompanyResearchTool(BaseTool):
    name: str = 'company_research'
    description: str = 'Research a company culture, mission, and products. Input: company name.'
    def _run(self, company: str) -> str:
        try:
            search = GoogleSerperAPIWrapper(serper_api_key=os.environ['SERPER_API_KEY'], type='search', k=3)
            return search.run(f'{company} company culture mission values')
        except Exception:
            return DuckDuckGoSearchRun().run(f'{company} company overview culture values')

class SkillGapAnalyzerTool(BaseTool):
    name: str = 'skill_gap_analyzer'
    description: str = 'Analyze gaps between candidate skills and job requirements. Input JSON: {"candidate_skills": [...], "job_requirements": [...]}'
    def _run(self, input_str: str) -> str:
        try:
            data = json.loads(input_str)
            cands = [c.lower().strip() for c in data.get('candidate_skills', [])]
            reqs  = [r.lower().strip() for r in data.get('job_requirements', [])]
            
            matched, missing = [], []
            for req in reqs:
                if any(cand in req or req in cand for cand in cands):
                    matched.append({"requirement": req, "status": "Matched"})
                else:
                    missing.append({"requirement": req, "status": "Missing/Gap"})
            
            pct = round(len(matched) / len(reqs) * 100, 1) if reqs else 0
            rec = 'Strong match! Apply immediately.' if pct >= 70 else 'Good match with minor gaps.' if pct >= 50 else 'Upskilling recommended.'
            return json.dumps({'overall_match_pct': pct, 'matched': matched, 'gaps': missing, 'recommendation': rec}, indent=2)
        except Exception as e:
            return f'Skill gap analyzer error: {e}'

class InterviewQuestionsTool(BaseTool):
    name: str = 'interview_questions'
    description: str = 'Fetch common interview questions for a role. Input: role name.'
    def _run(self, role: str) -> str:
        try:
            search = GoogleSerperAPIWrapper(serper_api_key=os.environ['SERPER_API_KEY'], type='search', k=3)
            return search.run(f'{role} interview questions technical behavioral')
        except Exception:
            return DuckDuckGoSearchRun().run(f'{role} common interview questions')

# Instantiate tools
job_search_tool = JobSearchTool()
salary_tool = SalaryResearchTool()
company_tool = CompanyResearchTool()
skill_gap_tool = SkillGapAnalyzerTool()
interview_tool = InterviewQuestionsTool()

# ─── MAIN WEB UI LAYOUT ────────────────────────────────────────────────
st.title("🤖 Multi-Agent Job Search System")
st.markdown("Coordinated team of AI agents running via **CrewAI** & **Llama 3.3 (Groq)** to find targets and map application assets.")

# Profile Configuration Form Layout
with st.expander("👤 Edit Candidate Persona Profile", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name", value="Alex Johnson")
        location = st.text_input("Location", value="San Francisco, CA (Open to Remote)")
        target_role = st.text_input("Target Role Title", value="Senior Machine Learning Engineer")
        target_industry = st.text_input("Target Industries", value="AI/ML, FinTech, or HealthTech")
        experience_years = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)
    with col2:
        current_role = st.text_input("Current Position", value="ML Engineer at a Series B startup")
        salary_expectation = st.text_input("Target Compensation", value="230,000 base + equity")
        career_goal = st.text_input("Long Term Objective", value="ML Engineering Manager within 2 years")

    col3, col4 = st.columns(2)
    with col3:
        tech_skills_raw = st.text_area("Technical Stack (Comma Separated)", 
            value="Python, PyTorch, TensorFlow, Scikit-learn, HuggingFace Transformers, LLM fine-tuning, RAG systems, MLOps, AWS SageMaker, Docker, Kubernetes")
        projects_raw = st.text_area("Key Project Highlights (One per line)", 
            value="Real-time fraud detection model - reduced false positives by 34%, saving $2.1M annually\nRAG-based customer support chatbot using LangChain + GPT-4\nMLOps pipeline migration to Kubeflow")
    with col4:
        soft_skills_raw = st.text_area("Core Soft Competencies (Comma Separated)", 
            value="Technical leadership, Cross-functional collaboration, Mentoring junior engineers")
        certs_raw = st.text_area("Credentials & Certifications (Comma Separated)", 
            value="AWS Certified Machine Learning - Specialty, Google Professional ML Engineer")

# Re-assemble data object from input parameters dynamically
CANDIDATE_PROFILE = {
    'name': name, 'location': location, 'target_role': target_role, 'target_industry': target_industry,
    'experience_years': experience_years, 'current_role': current_role, 'salary_expectation': salary_expectation,
    'career_goal': career_goal,
    'technical_skills': [s.strip() for s in tech_skills_raw.split(',') if s.strip()],
    'soft_skills': [s.strip() for s in soft_skills_raw.split(',') if s.strip()],
    'certifications': [s.strip() for s in certs_raw.split(',') if s.strip()],
    'key_projects': [p.strip() for p in projects_raw.split('\n') if p.strip()]
}

if st.button("🚀 Execute Strategic Multi-Agent Pipeline Assembly", type="primary"):
    
    # Define Agents
    profile_analyst = Agent(
        role='Senior Career Profile Analyst',
        goal='Analyze the candidate background, extract key strengths, and define a precise career positioning strategy.',
        backstory='Veteran career strategist mapping candidate profiles.',
        tools=[skill_gap_tool], llm=llm, verbose=True, allow_delegation=False
    )

    job_researcher = Agent(
        role='Elite Job Market Intelligence Researcher',
        goal='Find top target open roles, map descriptions, and analyze compensation benchmarks.',
        backstory='Data-driven web intelligence engine running precise live searches.',
        tools=[job_search_tool, salary_tool, company_tool], llm=llm, verbose=True, allow_delegation=False
    )

    resume_specialist = Agent(
        role='ATS-Optimized Resume Tailoring Expert',
        goal='Craft optimized resume block matrices targeted to bypass automated filtration systems.',
        backstory='Enterprise technical recruiter modeling text structures.',
        tools=[skill_gap_tool], llm=llm, verbose=True, allow_delegation=False
    )

    cover_letter_writer = Agent(
        role='Persuasive Cover Letter Storytelling Expert',
        goal='Produce compelling cover letters linking candidate histories to targeted missions.',
        backstory='Master copywriter skilled at drafting highly converting application materials.',
        tools=[company_tool], llm=creative_llm, verbose=True, allow_delegation=False
    )

    interview_coach = Agent(
        role='Executive Interview Preparation Coach',
        goal='Formulate targeted technical/behavioral preparation frameworks and negotiation scripts.',
        backstory='Elite hiring director leading candidate preparation execution loops.',
        tools=[interview_tool, company_tool], llm=llm, verbose=True, allow_delegation=False
    )

    report_compiler = Agent(
        role='Strategic Career Action Plan Compiler',
        goal='Consolidate prior multi-agent analytics outputs into a clear actionable blueprint markdown layout.',
        backstory='Operations officer tracking structural insights datasets.',
        tools=[], llm=llm, verbose=True, allow_delegation=True
    )

    # Define Tasks
    profile_str = json.dumps(CANDIDATE_PROFILE, indent=2)

    t1 = Task(description=f'Analyze data portfolio metrics:\n{profile_str}\nProvide strengths, UVPs, and placement strategy.', agent=profile_analyst, expected_output='Profile positioning analysis.')
    t2 = Task(description=f'Scan open listings for: {CANDIDATE_PROFILE["target_role"]}.\nProvide top options, matching scores, and pay scales.', agent=job_researcher, expected_output='Market research documentation data.')
    t3 = Task(description=f'Design an absolute elite ATS compliant parsing matrix block draft containing data highlights tailored to target parameters.', agent=resume_specialist, expected_output='ATS resume draft structural copy.')
    t4 = Task(description=f'Draft custom narrative cover letters connecting candidate strengths to targeted opportunities.', agent=cover_letter_writer, expected_output='Two clear cover letter document structures.')
    t5 = Task(description=f'Establish structured study guide containing sample custom technical questions, answer parameters, and strategic structural negotiation layouts.', agent=interview_coach, expected_output='Full behavioral interview prep playbook.')
    t6 = Task(description=f'Synthesize all prior outputs into a cohesive master markdown layout titled: `# Career Action Plan`. Make it actionable.', agent=report_compiler, expected_output='Master executive career roadmap dashboard dataset.')

    # Execution Loop
    with st.spinner("🧠 Orchestrating Agents... Running pipeline execution loops (Steps 1-6). Est time: 3 mins..."):
        crew = Crew(
            agents=[profile_analyst, job_researcher, resume_specialist, cover_letter_writer, interview_coach, report_compiler],
            tasks=[t1, t2, t3, t4, t5, t6],
            process=Process.sequential,
            verbose=True,
            max_rpm=30
        )
        final_result = crew.kickoff()
        
    st.success("✅ Execution Cycle Concluded Successfully!")
    st.markdown("---")
    st.subheader("📊 Output Strategic Action Matrix Blueprint")
    st.markdown(str(final_result))
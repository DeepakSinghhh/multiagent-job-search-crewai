# ⚡ NEXUS JOB AI
### Multi-Agent AI-Powered Job Search & Career Optimization Platform

A futuristic multi-agent system built using **CrewAI**, **LangChain**, **Groq LLM**, and **FAISS** that automates the entire job application workflow—from job discovery to resume optimization, cover letter generation, interview preparation, and skill-gap analysis.

---

## 🚀 Features

### 🤖 6 Specialized AI Agents

| Agent | Responsibility |
|---------|---------------|
| Profile Analyst | Analyzes candidate profile, strengths, and career positioning |
| Job Researcher | Searches relevant jobs and market opportunities |
| Resume Specialist | Creates ATS-optimized resumes tailored to jobs |
| Cover Letter Writer | Generates personalized cover letters |
| Interview Coach | Produces interview questions, STAR answers, and preparation guides |
| Report Compiler | Creates a complete career action plan |

---

## 🏗 System Architecture

```text
Candidate Profile
        │
        ▼
Profile Analyst
        │
        ▼
Job Researcher
        │
        ▼
Resume Specialist
        │
        ▼
Cover Letter Writer
        │
        ▼
Interview Coach
        │
        ▼
Report Compiler
        │
        ▼
Career Optimization Report
```

---

## 🛠 Tech Stack

### AI & Machine Learning
- CrewAI
- LangChain
- Groq (Llama 3.3 70B)
- Sentence Transformers
- FAISS

### Search & Data
- Serper API
- Real-time Web Search

### Frontend
- Streamlit
- Cyberpunk / Sci-Fi UI
- Custom CSS

### Visualization
- Matplotlib
- Skill Gap Analysis Charts

---

## 📊 Key Capabilities

✅ Automated Job Search

✅ ATS Resume Optimization

✅ Personalized Cover Letter Generation

✅ Interview Preparation

✅ Salary Negotiation Guidance

✅ Skill Gap Analysis

✅ Career Roadmap Generation

✅ Real-Time Job Market Research

---

## 🎨 Cyberpunk User Interface

The application features a futuristic design inspired by:

- Cyberpunk 2077
- Terminal Interfaces
- Retro Futurism
- Neon UI Systems

### UI Highlights

- Neon Cyan & Purple Theme
- Sci-Fi Dashboard
- Custom Agent Status Panels
- Animated Progress Tracking
- Terminal-Style Components
- Futuristic Typography

---

## 📁 Project Structure

```text
multiagent-job-search/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
│
├── outputs/
│   ├── resume.md
│   ├── cover_letter.md
│   ├── interview_prep.md
│   └── career_report.md
│
└── assets/
```

---

## ⚙ Installation

### Clone Repository

```bash
git clone https://github.com/DeepakSinghhh/multiagent-job-search-crewai.git

cd multiagent-job-search-crewai
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key
```

---

## ▶ Run Locally

```bash
streamlit run app.py
```

---

## ☁ Deploy on Streamlit Cloud

### 1. Push to GitHub

```bash
git add .
git commit -m "Initial Commit"
git push
```

### 2. Deploy

- Open Streamlit Cloud
- Connect GitHub Repository
- Select `app.py`
- Deploy

### 3. Add Secrets

```toml
GROQ_API_KEY = "your_key"
SERPER_API_KEY = "your_key"
```

---

## 📈 Skill Gap Analysis

The platform performs semantic comparison between:

- Candidate Skills
- Job Requirements

Using:

- Sentence Transformers
- Vector Embeddings
- FAISS Similarity Search

This helps identify:

- Missing Skills
- Learning Priorities
- Career Growth Opportunities

---

## 🎯 Example Output

The system generates:

### Resume
- ATS Optimized
- Job Specific
- Keyword Enhanced

### Cover Letter
- Personalized
- Recruiter Focused

### Interview Guide
- Technical Questions
- Behavioral Questions
- STAR Responses

### Career Report
- Skill Gaps
- Salary Insights
- Recommended Opportunities

---

## 💡 Future Improvements

- LinkedIn Integration
- Resume PDF Export
- Multi-LLM Support
- Job Recommendation Scoring
- Agent Memory
- RAG-Based Career Knowledge Base
- Dashboard Analytics

---

## 👨‍💻 Author

Deepak Kumar Singh

LinkedIn: https://www.linkedin.com/in/deepak-kumar-singh-698b02350/

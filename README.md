◽ Multi-Agent Job Search System

Automated job search pipeline using 6 AI agents built with CrewAI and LangChain.



🔹 Architecture
6 Specialized Agents: Profile Analyst, Job Researcher, Resume Specialist, Cover Letter Writer, Interview Coach, Report Compiler

Real-time Search: Live job search integration via Serper API

Skill-Gap Analysis: Semantic skill-gap analysis using FAISS + SentenceTransformers

Core LLM: llama-3.3-70b-versatile via Groq

🔹 Tech Stack

CrewAI · LangChain · Groq · FAISS · SentenceTransformers · Matplotlib



🔹 Quick Start

Open in Google Colab (badge below)
Add Groq + Serper API keys

Edit CANDIDATE_PROFILE with your info
Run all cells



🔹 Results

Finds top 5 matching jobs with fit scores

Generates ATS-optimized resume

Writes personalized cover letters

Full interview prep guide + salary negotiation script
# RAG-GenAI-Insurance-Claims-Tool
 GenAI-RAG Insurance Claims Assistant
An intelligent, Generative AI-powered insurance claims processing dashboard built with Streamlit, LangChain, OpenAI, and FAISS. The tool uses Retrieval-Augmented Generation (RAG) to analyze and reason over multiple insurance documents, enabling faster and more accurate decisions across various stages of claims processing.

🚀 Features
📌 Multi-functional Claims Dashboard
A clean and interactive Streamlit interface with the following modules:

Liability Decision: Analyzes claim forms, policies, police reports, and traffic laws to suggest liability with justifications.

Eligibility Check: Assesses if policy coverage aligns with repair and medical bills, identifying mismatches or overcharges.

Document Summary: Summarizes key insights and extracts data points from uploaded documents.

Q&A System: Ask questions related to uploaded documents and get precise, context-aware answers.

⚙️ How It Works
🧠 Powered by GenAI + RAG
Uses LangChain's multi-query retrieval and stuffing chains.

Powered by gpt-4o-mini from OpenAI for both classification and contextual reasoning.

Uses PDF parsing, document chunking, and FAISS vector store to retrieve relevant content.

Each module runs a domain-specific RAG pipeline tailored for the task (e.g., liability reasoning or eligibility validation).

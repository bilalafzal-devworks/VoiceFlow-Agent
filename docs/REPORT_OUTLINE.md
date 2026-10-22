# Project report outline: VoiceFlow Agent
1. **Problem statement**: typing queries and manually copying answers into documents is slow; plain LLMs hallucinate; users want hands-free, trustworthy answers with delivery to Google Workspace.
2. **Methodology**: iterative build (STT → RAG → agents → automation), free-tier stack (Streamlit, Groq, Hugging Face, ChromaDB, DuckDuckGo, n8n), evaluation with a fixed query set.
3. **Agent design**: roles of Router, Retriever, Researcher, Writer, Reviewer; why tools are run in plain Python and agents reason over results; Writer ⇄ Reviewer loop (max 2 revisions); data flow diagram.
4. **Prompt design**: structure (role, goal, constraints, format, fallback); few-shot Router; JSON contracts; guardrails (grounding, citations, "I don't know", approval gate, injection resistance); v1 vs v2 differences.
5. **RAG and automation design**: chunking (800/100 vs 200), MiniLM embeddings, cosine search, top-k=4; n8n nodes and error branch; double approval check.
6. **Results**: the `eval_results.md` table (accuracy, groundedness, format compliance for v1 vs v2), example traces, screenshots, observed failures.
7. **Limitations**: rate limits, ephemeral storage on Streamlit Cloud, no auth, OCR gap, MiniLM 256-token window, Whisper errors, DuckDuckGo variability.
8. **Future work**: authentication, hosted vector DB, reranking, OCR, streaming answers, conversation memory, text-to-speech replies, more n8n actions (Calendar, Drive folders).
9. **References and appendix**: prompt files, workflow JSON, test plan.

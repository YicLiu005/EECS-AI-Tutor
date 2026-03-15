#  KidLearnAI: Agentic Teaching Assistant

**KidLearnAI** is a multi-modal, agentic teaching assistant designed for elementary school education. It goes beyond simple question-answering by providing pedagogical scaffolding, step-by-step guidance, and strict student evaluation using Vision-Language Models (VLMs) and Retrieval-Augmented Generation (RAG).

##  Key Features

* ** Dual-Mode Architecture:** 
  * **Learning Mode:** Acts as a patient tutor, explaining concepts step-by-step and grounding answers in retrieved curriculum facts.
  * **Exam Mode:** Acts as a strict evaluator, grading student answers (text or handwritten images) with a structured rubric and providing actionable feedback.
* ** Multi-Modal Vision Processing:** Students can upload images of handwritten worksheets or math problems. The system forces the VLM to accurately extract text and math formulas before processing the query, preventing multimodal hallucinations.
* ** Retrieval-Augmented Generation (RAG):** Uses local vector search to ground the AI's answers in curriculum-aligned knowledge. Includes a "Knowledge Injector" feature to test RAG capabilities dynamically.
* ** Generate-and-Score Pipeline:** Instead of zero-shot answering, the system generates multiple candidate answers and deterministically scores them to output the safest and most structured response.

##  Project Structure

* `web_ui.py`: The main Streamlit frontend interface and logic router.
* `rag_module.py`: Handles document chunking, semantic embedding, and local vector retrieval.
* `gemini_adapter.py`: A unified wrapper for Google's Gemini API (Text, Vision, and Embedding).
* `answer_module.py` & `answer_score_module.py`: The Generate-and-Score pipeline for Learning Mode.
* `student_grade_module.py`: LLM-as-a-Judge logic for evaluating student answers in Exam Mode.
* `learning_module.py` & `schemas.py`: Backend modules for data structures and logging interactions.

##  How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/EECS-AI-Tutor.git
   cd EECS-AI-Tutor

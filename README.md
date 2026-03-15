# EECS-AI-Tutor
An agentic AI tutor with RAG and multimodal capabilities for elementary students.
## 📂 Project Structure

* `web_ui.py`: The main Streamlit frontend interface.
* `rag_module.py`: Handles document chunking, semantic embedding, and retrieval.
* `gemini_adapter.py`: A unified wrapper for Google's Gemini 2.5 Flash API (Text, Vision, and Embedding).
* `answer_module.py` & `answer_score_module.py`: The Generate-and-Score pipeline for Learning Mode.
* `student_grade_module.py`: LLM-as-a-Judge logic for evaluating student answers in Exam Mode.

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/KidLearnAI.git
   cd KidLearnAI

  #Install dependencies:
  pip install -r requirements.txt
  #Set your API Key:
   Open config.py (or web_ui.py) and replace the GEMINI_API_KEY placeholder with your valid Google Gemini API key.
   #Run the Streamlit App:
   python -m streamlit run web_ui.py

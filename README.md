KidLearnAI

KidLearnAI is a simple AI tutoring assistant designed for elementary school learning.
The system supports both text questions and handwritten image inputs, and generates step-by-step explanations using Large Language Models and Retrieval-Augmented Generation (RAG).

The interface is built with Streamlit, allowing users to interact with the tutor through a web UI.

Project Structure
KidLearnAI/
│
├── learning_doc/        # Textbook or learning materials used as the knowledge base
├── storage/             # Local vector database for RAG retrieval
│
├── web_ui.py            # Streamlit web interface
├── main.py              # Backend entry point
│
├── rag_module.py        # Handles document chunking and retrieval
├── answer_module.py     # Generates structured answers
├── gemini_adapter.py    # Wrapper for Gemini API
├── schemas.py           # Shared data structures
├── config.py            # Configuration settings

How to Run
1. Install dependencies

Make sure you have Python 3.9+ installed.

pip install -r requirements.txt

2. Start the Web UI
Open the web_ui.py

Run the Streamlit interface:

streamlit run web_ui.py

Then open the browser:

http://localhost:8501

Example usage
1.In Learning mode, add question for example like A rectangular prism has dimensions 6 cm × 4 cm × 9 cm. What is its volume in cubic centimeters? in the Input box, then the system will output answer and explanation together or upload the question img then typing the question.
2.In Exam mode, choose the grade level and subject, then click start exam, the system will generate question for user.

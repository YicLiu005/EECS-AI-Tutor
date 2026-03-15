# web_ui.py
# Streamlit interface for the tutor app.

import streamlit as st
import os
from PIL import Image

# ==========================================
# Page setup
# ==========================================
st.set_page_config(page_title="Pro AI Tutor", page_icon="🎓", layout="wide")
os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY_HERE"

# ==========================================
# Load backend modules
# ==========================================
from rag_module import RAGModule
from answer_module import AnswerModule
from answer_score_module import AnswerScoringModule
from student_grade_module import StudentGradingModule
from gemini_adapter import GeminiAdapter


# Cache backend modules
@st.cache_resource
def load_backend():
    gemini = GeminiAdapter()
    rag = RAGModule(storage_dir="storage")
    answerer = AnswerModule()
    scorer = AnswerScoringModule(use_llm_judge=False)
    grader = StudentGradingModule(rag=rag)
    return gemini, rag, answerer, scorer, grader


try:
    gemini, rag, answerer, scorer, grader = load_backend()
except Exception as e:
    st.error(f"Failed to load backend modules: {e}\nPlease ensure all module files (.py) are in the same folder.")
    st.stop()

# ==========================================
# Session state
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_exam_question" not in st.session_state:
    st.session_state.current_exam_question = None

# ==========================================
# Sidebar
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3074/3074368.png", width=80)
    st.title("Control Panel")
    st.markdown("---")

    st.subheader("🎓 Select Mode")
    current_mode = st.radio(
        "Choose how the AI interacts with you:",
        options=["📖 Learning Mode (Step-by-step)", "📝 Exam Mode (Strict Grading)"],
        index=0
    )

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_exam_question = None
        st.rerun()

    st.markdown("---")
    st.caption("Powered by KidLearnAI Backend & Gemini")

# ==========================================
# Main page
# ==========================================
st.title("🍎 Elementary AI Tutor")
if "Learning" in current_mode:
    st.info(
        "💡 **Learning Mode Active:** Ask me any question. I will find relevant knowledge and explain it step-by-step!")
    st.session_state.current_exam_question = None
else:
    st.warning(
        "⏱️ **Exam Mode Active:** Click 'Start Exam' to get a question, then type your answer below to be graded.")
    is_expanded = (st.session_state.current_exam_question is None)
    with st.expander("⚙️ Exam Settings & Start", expanded=is_expanded):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            selected_grade = st.selectbox("Grade Level",
                                          ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Grade 6"])
        with col2:
            selected_subject = st.selectbox("Subject", ["Math", "English", "Science", "Social Studies"])
        with col3:
            st.write("")
            st.write("")
            start_exam_btn = st.button("🚀 Start Exam", use_container_width=True)

        if start_exam_btn:
            st.session_state.messages = []
            prompt = f"You are a strict examiner. Generate ONE {selected_subject} question suitable for a {selected_grade} student. Do NOT provide the answer or hints. Just ask the question directly."
            with st.spinner("Generating exam question..."):
                try:
                    q_text = gemini.generate_text(prompt, temperature=0.7)
                    st.session_state.current_exam_question = q_text
                    st.session_state.messages.append({"role": "assistant", "content": q_text})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating question: {e}")

with st.expander("📎 Attach an image (Worksheet or Handwriting)", expanded=False):
    uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"], label_visibility="hidden")
    image_bytes = None
    mime_type = "image/png"
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        image_bytes = uploaded_file.getvalue()
        mime_type = uploaded_file.type

for message in st.session_state.messages:
    avatar_icon = "🧑‍🎓" if message["role"] == "user" else "🍎"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

# ==========================================
# Chat logic
# ==========================================
if prompt := st.chat_input("Type your message here..."):
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🍎"):
        message_placeholder = st.empty()

        with st.spinner("Processing through AI Engine..."):
            try:
                if "Learning" in current_mode:
                    problem_text = prompt
                    if image_bytes and not problem_text.strip():
                        spec = rag.parse_problem_image(image_bytes, mime_type=mime_type)
                        problem_text = spec.problem_text

                    if not problem_text.strip():
                        st.warning("Please type a question or upload a readable image.")
                        st.stop()

                    contexts = rag.retrieve(problem_text, top_k=5)

                    candidates = answerer.generate_candidates(
                        problem_text=problem_text,
                        contexts=contexts,
                        style="detailed"
                    )

                    selection = scorer.select_best(
                        problem_text=problem_text,
                        contexts=contexts,
                        candidates=candidates
                    )
                    best = selection.best_answer

                    reply = f"**Final Answer:**\n{best.final_answer}\n\n**Steps:**\n{best.steps}"
                    message_placeholder.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

                else:
                    exam_q = st.session_state.current_exam_question
                    if not exam_q:
                        exam_q = "Please grade the student's input."
                        if len(st.session_state.messages) >= 2:
                            exam_q = st.session_state.messages[-2]["content"]

                    contexts = rag.retrieve(exam_q, top_k=3)

                    grade_result = grader.grade_student(
                        problem_text=exam_q,
                        contexts=contexts,
                        student_answer_text=prompt,
                        student_answer_image_bytes=image_bytes,
                        student_answer_image_mime=mime_type
                    )

                    status_emoji = "✅" if grade_result.is_correct == "correct" else "❌" if grade_result.is_correct == "incorrect" else "⚠️"

                    reply = f"### Grade: {grade_result.score}/100 {status_emoji}\n\n"
                    reply += f"**Teacher's Feedback:**\n{grade_result.feedback}\n\n"

                    details = "\n".join(
                        [f"- **{k.capitalize()}:** {v}/100" for k, v in grade_result.rubric_breakdown.items()])
                    reply += f"<details><summary>📋 View Grading Rubric Breakdown</summary>\n\n{details}\n</details>"

                    message_placeholder.markdown(reply, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

                    st.session_state.current_exam_question = None

            except Exception as e:
                st.error(f"System Backend Error: {str(e)}")
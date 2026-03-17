import streamlit as st
from PIL import Image
import re

from rag_module import RAGModule
from answer_module import AnswerModule
from gemini_adapter import GeminiAdapter


st.set_page_config(page_title="Pro AI Tutor", page_icon="🎓", layout="wide")
if "cache_bust" not in st.session_state:
    st.session_state.cache_bust = "v1"
@st.cache_resource
def load_backend(cache_bust: str):
    gemini = GeminiAdapter()
    rag = RAGModule(storage_dir="storage")
    answerer = AnswerModule(
        textbook_md_dir=r"C:\Users\admin\Desktop\EECSa\learning doc"
    )
    return gemini, rag, answerer


try:
    gemini, rag, answerer = load_backend(st.session_state.cache_bust)
except Exception as e:
    st.error(
        f"Failed to load backend modules: {e}\n"
        f"Please ensure all module files (.py) are in the same folder."
    )
    st.stop()
if "learning_messages" not in st.session_state:
    st.session_state.learning_messages = []

if "exam_messages" not in st.session_state:
    st.session_state.exam_messages = []

if "current_exam_question" not in st.session_state:
    st.session_state.current_exam_question = None

if "answer_style" not in st.session_state:
    st.session_state.answer_style = "detailed"  # "brief" | "detailed"

if "learning_followup" not in st.session_state:
    st.session_state.learning_followup = {
        "active": False,
        "problem_text": "",
        "last_answer": ""
    }
if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None

if "uploaded_image_mime" not in st.session_state:
    st.session_state.uploaded_image_mime = "image/png"

if "uploaded_image_name" not in st.session_state:
    st.session_state.uploaded_image_name = ""
def highlight_citations(text: str) -> str:
    """
    Highlight citation blocks in Learning Mode.

    New supported format:
      <cite chunk_id="..."> ... </cite>

    Backward compatible:
      - [CITE:...]
      - CITE:... (no brackets)
    """
    if not text:
        return text

    def repl_cite_block(match):
        chunk_id = (match.group(1) or "").strip()
        content = (match.group(2) or "").strip()
        return (
            f'<mark title="{chunk_id}" '
            f'style="background-color:#fff3a3; padding:2px 4px; border-radius:4px;">'
            f'{content}'
            f'</mark>'
        )

    # New format: <cite chunk_id="..."> ... </cite>
    text = re.sub(
        r'<cite\s+chunk_id="([^"]+)">(.*?)</cite>',
        repl_cite_block,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Backward compatibility for old format
    text = re.sub(r"\[CITE:([^\]]+)\]", r'<mark>[CITE:\1]</mark>', text)
    text = re.sub(r"(CITE:[^\s\)\]]+)", r'<mark>\1</mark>', text)

    return text


def format_citations(citations: list[str]) -> str:
    if not citations:
        return "_No citations._"

    items = []
    for cid in citations:
        info = AnswerModule.describe_citation(cid)
        if info["type"] == "textbook":
            items.append(f"- **Textbook**: {info['label']}\n  - `{info['chunk_id']}`")
        else:
            items.append(f"- `{info['chunk_id']}`")
    return "\n".join(items)


def build_assistant_reply(best) -> str:
    cite_text = format_citations(best.citations)
    return (
        f"**FINAL_ANSWER:**\n{best.final_answer}\n\n"
        f"**STEPS:**\n{best.steps}\n\n"
        f"**SOURCES (CITATIONS):**\n{cite_text}"
    )


def normalize_mime_type(mime_type: str) -> str:
    mime = (mime_type or "image/png").lower().strip()
    allowed = {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
        "image/bmp",
        "image/gif",
    }
    if mime not in allowed:
        return "image/png"
    if mime == "image/jpg":
        return "image/jpeg"
    return mime


def extract_problem_and_request(user_prompt: str, image_bytes: bytes | None, mime_type: str) -> tuple[str, str]:
    """
    Returns:
      retrieval_problem_text: 
      final_problem_text:    
    """
    user_prompt = (user_prompt or "").strip()

  
    if image_bytes:
        spec = rag.parse_problem_image(
            image_bytes,
            mime_type=normalize_mime_type(mime_type)
        )
        image_problem_text = (spec.problem_text or "").strip()

        if not image_problem_text:
            raise ValueError("Failed to extract problem text from the uploaded image.")

        if user_prompt:
            final_problem_text = (
                f"Problem:\n{image_problem_text}\n\n"
                f"Student request:\n{user_prompt}"
            )
        else:
            final_problem_text = image_problem_text

        return image_problem_text, final_problem_text


    if user_prompt:
        return user_prompt, user_prompt

    raise ValueError("Please type a question or upload a readable image.")


def clear_uploaded_image():
    st.session_state.uploaded_image_bytes = None
    st.session_state.uploaded_image_mime = "image/png"
    st.session_state.uploaded_image_name = ""

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3074/3074368.png", width=80)
    st.title("Control Panel")
    st.markdown("---")

    st.subheader("🎓 Select Mode")
    current_mode = st.radio(
        "Choose how the AI interacts with you:",
        options=["📖 Learning Mode (Step-by-step)", "📝 Exam Mode"],
        index=0
    )

    st.markdown("---")
    if st.button("🔄 Reload Textbooks (refresh cache)", use_container_width=True):
        old = st.session_state.cache_bust
        if old.startswith("v") and old[1:].isdigit():
            st.session_state.cache_bust = f"v{int(old[1:]) + 1}"
        else:
            st.session_state.cache_bust = "v2"
        st.rerun()

    chunks_loaded = len(getattr(answerer, "_textbook_chunks", []))
    files_loaded = len({c.metadata.get("file") for c in getattr(answerer, "_textbook_chunks", [])})
    st.caption(f"Cache version: {st.session_state.cache_bust}")
    st.caption(f"Textbook files loaded: {files_loaded}")
    st.caption(f"Textbook chunks loaded: {chunks_loaded}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History (Current Mode)", use_container_width=True):
        if "Learning" in current_mode:
            st.session_state.learning_messages = []
            st.session_state.learning_followup = {
                "active": False,
                "problem_text": "",
                "last_answer": ""
            }
            clear_uploaded_image()
        else:
            st.session_state.exam_messages = []
            st.session_state.current_exam_question = None
        st.rerun()

    st.markdown("---")
    st.caption("Powered by KidLearnAI Backend & Gemini")

# Main UI
st.title("🍎 Elementary AI Tutor")

is_learning = "Learning" in current_mode
is_exam = "Exam" in current_mode

if is_learning:
    st.info("💡 **Learning Mode Active:** Ask me any question. I will retrieve knowledge and explain step-by-step!")
else:
    st.warning("⏱️ **Exam Mode Active:** Click 'Start Exam' to get a question, then answer it. AI will give a reference answer (not grading).")
    st.session_state.learning_followup["active"] = False

if is_learning:
    st.session_state.answer_style = st.radio(
        "Answer Style",
        options=["detailed", "brief"],
        index=0,
        horizontal=True
    )
# Exam panel

if is_exam:
    is_expanded = (st.session_state.current_exam_question is None)
    with st.expander("⚙️ Exam Settings & Start", expanded=is_expanded):
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            selected_grade = st.selectbox(
                "Grade Level",
                ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Grade 6"]
            )

        with col2:
            selected_subject = st.selectbox(
                "Subject",
                ["Math", "English", "Science", "Social Studies"]
            )

        with col3:
            st.write("")
            st.write("")
            start_exam_btn = st.button("🚀 Start Exam", use_container_width=True)

        if start_exam_btn:
            st.session_state.exam_messages = []
            prompt = (
                f"You are a strict examiner. Generate ONE {selected_subject} question suitable for a "
                f"{selected_grade} student. Do NOT provide the answer or hints. Just ask the question directly."
            )
            with st.spinner("Generating exam question..."):
                try:
                    q_text = gemini.generate_text(prompt, temperature=0.7)
                    st.session_state.current_exam_question = q_text
                    st.session_state.exam_messages.append({"role": "assistant", "content": q_text})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating question: {e}")

# Image upload
with st.expander("📎 Attach an image (Worksheet or Handwriting)", expanded=False):
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["png", "jpg", "jpeg"],
        label_visibility="hidden",
        key="main_image_uploader"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)

            current_bytes = uploaded_file.getvalue()
            current_mime = normalize_mime_type(uploaded_file.type)
            current_name = uploaded_file.name or ""

            
            st.session_state.uploaded_image_bytes = current_bytes
            st.session_state.uploaded_image_mime = current_mime
            st.session_state.uploaded_image_name = current_name

        except Exception as e:
            st.error(f"Cannot read uploaded image: {e}")
            clear_uploaded_image()


    elif st.session_state.uploaded_image_bytes is not None:
        try:
            st.image(
                st.session_state.uploaded_image_bytes,
                caption=f"Uploaded Image ({st.session_state.uploaded_image_name})" if st.session_state.uploaded_image_name else "Uploaded Image",
                width=300
            )
        except Exception:
            pass

    if st.session_state.uploaded_image_bytes is not None:
        if st.button("❌ Remove uploaded image", use_container_width=True):
            clear_uploaded_image()
            st.rerun()


image_bytes = st.session_state.uploaded_image_bytes
mime_type = st.session_state.uploaded_image_mime
# Render messages per mode

messages_to_render = st.session_state.learning_messages if is_learning else st.session_state.exam_messages
for message in messages_to_render:
    avatar_icon = "🧑‍🎓" if message["role"] == "user" else "🍎"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"], unsafe_allow_html=True)

# Learning followup buttons

if is_learning and st.session_state.learning_followup["active"]:
    st.markdown("---")
    st.caption("Did that explanation help?")
    c1, c2 = st.columns(2)

    if c1.button("😵 Still confuse", use_container_width=True):
        problem_text = st.session_state.learning_followup["problem_text"].strip()
        last_answer = st.session_state.learning_followup["last_answer"].strip()

        followup_request = (
            "The student is still confused. Please explain again with:\n"
            "1) simpler words,\n"
            "2) a very small worked example,\n"
            "3) common mistakes to avoid,\n"
            "4) a quick check question at the end.\n\n"
            f"Original question:\n{problem_text}\n\n"
            f"Previous answer:\n{last_answer}\n"
        )

        with st.spinner("Explaining again..."):
            try:
                contexts = rag.retrieve(problem_text, top_k=5)
                candidates = answerer.generate_candidates(
                    problem_text=followup_request,
                    contexts=contexts,
                    style="detailed",
                    history=st.session_state.learning_messages,
                )
                best = candidates[0]

                reply = "**Let me explain again (simpler):**\n\n" + build_assistant_reply(best)
                reply = highlight_citations(reply)
                st.session_state.learning_messages.append({"role": "assistant", "content": reply})

                st.session_state.learning_followup["last_answer"] = best.raw_model_output
                st.rerun()

            except Exception as e:
                st.error(f"System Backend Error: {str(e)}")

    if c2.button("✅ I got it", use_container_width=True):
        st.session_state.learning_followup = {
            "active": False,
            "problem_text": "",
            "last_answer": ""
        }
        st.session_state.learning_messages.append({
            "role": "assistant",
            "content": "Awesome — glad you got it! 👍"
        })
        st.rerun()
# Input route

chat_placeholder = "Ask a follow-up question about the uploaded image, or type a question..."
if prompt := st.chat_input(chat_placeholder):
    if is_learning:
        st.session_state.learning_messages.append({"role": "user", "content": prompt})
    else:
        st.session_state.exam_messages.append({"role": "user", "content": prompt})

    with st.spinner("Processing through AI Engine..."):
        try:
            if is_learning:
                retrieval_problem_text, final_problem_text = extract_problem_and_request(
                    user_prompt=prompt,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                )

                contexts = rag.retrieve(retrieval_problem_text, top_k=5)
                answer_style = st.session_state.answer_style

                candidates = answerer.generate_candidates(
                    problem_text=final_problem_text,
                    contexts=contexts,
                    style=answer_style,
                    history=st.session_state.learning_messages,
                )
                best = candidates[0]

                reply = f"**Answer ({answer_style}):**\n\n" + build_assistant_reply(best)
                reply = highlight_citations(reply)
                st.session_state.learning_messages.append({"role": "assistant", "content": reply})

                st.session_state.learning_followup = {
                    "active": True,
                    "problem_text": retrieval_problem_text,
                    "last_answer": best.raw_model_output,
                }
                st.rerun()

            else:
                exam_q = st.session_state.current_exam_question
                if not exam_q:
                    exam_q = "Please answer the exam question based on the context."

                contexts = rag.retrieve(exam_q, top_k=3)

                candidates = answerer.generate_candidates(
                    problem_text=exam_q,
                    contexts=contexts,
                    style="detailed",
                    history=st.session_state.exam_messages,
                )
                best = candidates[0]

                reply = "### Reference Answer\n\n" + build_assistant_reply(best)
                st.session_state.exam_messages.append({"role": "assistant", "content": reply})

                st.session_state.current_exam_question = None
                st.rerun()

        except Exception as e:
            st.error(f"System Backend Error: {str(e)}")
# app.py
import streamlit as st
import os
import json
import bcrypt
import whisper
import tempfile
import fitz
from core import (
    init_user_storage, process_pdf, process_video,
    process_youtube, ask_question, clear_user_knowledge_base
)

USERS_FILE = "users.json"

def load_users():
    return json.load(open(USERS_FILE)) if os.path.exists(USERS_FILE) else {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

st.set_page_config(page_title="Knowledge Grounding App", layout="wide")

if "user" not in st.session_state:
    st.session_state.user = None
if "mode" not in st.session_state:
    st.session_state.mode = "Login"

users = load_users()

# 🔐 LOGIN / SIGNUP
if not st.session_state.user:
    col1, col2, col3 = st.columns([2, 4, 2])
    with col2:
        st.markdown("### 🔐 Login or Signup")

        if st.session_state.mode == "Login":
            st.subheader("🔓 Login")
            username = st.text_input("User ID", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                if username in users and verify_password(password, users[username]["password"]):
                    st.session_state.user = username
                    init_user_storage(username)
                    st.success(f"Welcome, {users[username]['name']}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            st.button("Go to Signup", on_click=lambda: st.session_state.update(mode="Signup"))

        else:
            st.subheader("📝 Signup")
            name = st.text_input("Full Name", key="signup_name")
            new_user = st.text_input("User ID", key="signup_user")
            new_pass = st.text_input("Password", type="password", key="signup_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", key="signup_confirm")
            if st.button("Signup"):
                if not all([name, new_user, new_pass, confirm_pass]):
                    st.error("All fields are required.")
                elif new_user in users:
                    st.error("User ID already exists.")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match.")
                else:
                    users[new_user] = {"name": name, "password": hash_password(new_pass)}
                    save_users(users)
                    st.success("✅ Signup successful. You can now log in.")
                    st.session_state.mode = "Login"
                    st.rerun()
            st.button("Go to Login", on_click=lambda: st.session_state.update(mode="Login"))
    st.stop()

# ✅ MAIN APP
username = st.session_state.user
display_name = users[username]["name"]

st.sidebar.title(f"👋 Welcome, {display_name}")
if st.sidebar.button("🧹 Clear My Knowledge Base"):
    clear_user_knowledge_base(username)
    st.sidebar.success("✅ Knowledge base cleared.")
if st.sidebar.button("🚪 Logout"):
    st.session_state.user = None
    st.rerun()

st.title("📚 Knowledge Grounding from PDF / Video / YouTube")
tab1, tab2 = st.tabs(["📤 Upload / Paste", "💬 Ask a Question"])

# TAB 1
with tab1:
    st.subheader("Upload PDF or MP4, or paste YouTube link")
    extracted_text = ""

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "mp4"])
    youtube_link = st.text_input("Or paste YouTube URL")

    if uploaded_file:
        raw_dir = os.path.join("data", "users", username, "raw_text")
        os.makedirs(raw_dir, exist_ok=True)
        file_path = os.path.join(raw_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        if uploaded_file.name.endswith(".pdf"):
            st.info("📄 Processing PDF...")
            process_pdf(file_path, username)
            doc = fitz.open(file_path)
            extracted_text = " ".join([page.get_text() for page in doc])
            st.success("✅ PDF processed.")
        elif uploaded_file.name.endswith(".mp4"):
            st.info("🎥 Transcribing video...")
            process_video(file_path, username)
            model = whisper.load_model("base")
            result = model.transcribe(file_path)
            extracted_text = result["text"]
            st.success("✅ Video processed.")

    elif youtube_link:
        st.info("🔗 Processing YouTube link...")
        process_youtube(youtube_link, username)
        audio_path = os.path.join(tempfile.gettempdir(), "yt_video.wav")
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        extracted_text = result["text"]
        st.success("✅ YouTube processed.")

    if extracted_text:
        with st.expander("📝 View Extracted Text"):
            st.markdown(extracted_text)

# TAB 2
with tab2:
    st.subheader("Ask a question based on all your uploaded content")
    question = st.text_input("Ask here:")
    if question:
        answer, cosine_score, gemini_score = ask_question(question, username)
        st.markdown("### 🧠 Answer")
        st.success(answer)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Similarity Score", f"{cosine_score:.2f}%")
            with st.expander("ℹ️ How it's calculated"):
                st.info("""
**Similarity Score** is based on cosine similarity between your question and the most relevant document chunk.
- 100% = exact semantic match
- Lower = less relevant
""")
       

import os
import fitz
import whisper
import faiss
import numpy as np
import json
from yt_dlp import YoutubeDL
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile
import shutil

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
FFMPEG_PATH = r"C:\ffmpeg\bin"  # Adjust if needed

def init_user_storage(user_id):
    os.makedirs(f"data/users/{user_id}/raw_text", exist_ok=True)
    os.makedirs(f"data/users/{user_id}/faiss_index", exist_ok=True)

def clear_user_knowledge_base(user_id):
    user_dir = f"data/users/{user_id}/faiss_index"
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
    os.makedirs(user_dir)

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def embed_chunks(chunks):
    return embedding_model.encode(chunks).tolist()

def save_to_faiss(vectors, texts, user_id):
    index_path = f"data/users/{user_id}/faiss_index/index.faiss"
    doc_path = f"data/users/{user_id}/faiss_index/docs.txt"

    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        with open(doc_path, "r", encoding="utf-8") as f:
            existing_docs = f.read().split("||||")
    else:
        index = faiss.IndexFlatL2(len(vectors[0]))
        existing_docs = []

    index.add(np.array(vectors).astype("float32"))
    all_docs = existing_docs + texts

    faiss.write_index(index, index_path)
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("||||".join(all_docs))

def process_pdf(file_path, user_id):
    doc = fitz.open(file_path)
    text = " ".join([page.get_text() for page in doc])
    chunks = chunk_text(text)
    vectors = embed_chunks(chunks)
    save_to_faiss(vectors, chunks, user_id)

def process_video(file_path, user_id):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    chunks = chunk_text(result["text"])
    vectors = embed_chunks(chunks)
    save_to_faiss(vectors, chunks, user_id)

def process_youtube(link, user_id):
    tmp_dir = tempfile.gettempdir()
    audio_path = os.path.join(tmp_dir, "yt_video.wav")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(tmp_dir, "yt_video.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "ffmpeg_location": FFMPEG_PATH,
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])
    if not os.path.exists(audio_path):
        raise FileNotFoundError("❌ Audio file not found after yt_dlp run.")
    process_video(audio_path, user_id)

def ask_question(query, user_id):
    index_path = f"data/users/{user_id}/faiss_index/index.faiss"
    doc_path = f"data/users/{user_id}/faiss_index/docs.txt"

    if not os.path.exists(index_path) or not os.path.exists(doc_path):
        return "No documents found. Please upload something first.", 0, 0

    index = faiss.read_index(index_path)
    with open(doc_path, "r", encoding="utf-8") as f:
        docs = f.read().split("||||")

    q_vector = embedding_model.encode([query])
    D, I = index.search(np.array(q_vector).astype("float32"), k=3)
    retrieved = [docs[i] for i in I[0]]
    context = "\n".join(retrieved)

    llm = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
You are an assistant answering based only on the given context. Respond in JSON with your answer and a confidence score (0 to 100) with reason.

Context:
{context}

Question:
{query}

Respond ONLY in this format:
{{"answer": "your answer here", "confidence": 87 (reason)}}
"""

    response = llm.generate_content(prompt)
    try:
        result = json.loads(response.text)
        answer = result.get("answer", "").strip()
        gemini_confidence = float(result.get("confidence", 0))
    except Exception:
        answer = response.text.strip()
        gemini_confidence = 0.0

    top_vector = embedding_model.encode([retrieved[0]])
    cosine_score = cosine_similarity(q_vector, top_vector)[0][0] * 100

    return answer, cosine_score, gemini_confidence

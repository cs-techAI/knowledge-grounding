📚 Knowledge Grounding App
Ground answers from your own knowledge base — extracted from PDFs, MP4s, or YouTube videos — and query them using a conversational AI (Gemini + Semantic Search).
Supports login/signup with per-user isolated vector stores and scoring system.

📁 Folder Structure
bash
Copy
Edit
knowledge_grounding/
├── app.py                      # Streamlit UI app with login, upload, QA
├── core.py                     # Core processing logic: embeddings, vector search, LLM
├── users.json                  # Registered users with hashed passwords
├── .env                        # Environment file with GEMINI_API_KEY
├── data/
│   └── users/
│       └── <username>/
│           ├── raw_text/       # Original uploaded files
│           └── faiss_index/    # Vector index & context chunks
│               ├── index.faiss
│               └── docs.txt
├── requirements.txt            # Python package dependencies
🔧 Setup Instructions
Clone the repo

bash
Copy
Edit
git clone https://github.com/your-username/knowledge-grounding-app.git
cd knowledge_grounding-app
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Install FFmpeg

Download from https://ffmpeg.org/download.html

Add to core.py: update FFMPEG_PATH = r"C:\path\to\ffmpeg\bin"

Add .env file with your Gemini API key

ini
Copy
Edit
GEMINI_API_KEY=your_google_gemini_api_key
Run the app

bash
Copy
Edit
streamlit run app.py
🔐 Login & Signup
Each user registers with:

Full name

Unique User ID

Password + confirmation

Passwords are hashed securely using bcrypt.

After login:

Data is saved in a private folder: data/users/<user_id>/

Vector index and uploaded documents are not shared across users.

📂 Upload & Process
You can upload:

📄 PDF files → Extracted using PyMuPDF

🎥 MP4 video files → Transcribed using OpenAI Whisper

🔗 YouTube links → Downloaded with yt_dlp, audio extracted and transcribed

All content is:

🔍 Chunked

🔡 Embedded using sentence-transformers

💾 Stored in FAISS index per user

💬 Ask Questions
Ask anything grounded on your uploaded data.
The app:

Retrieves top-3 relevant chunks via FAISS

Sends context + query to Gemini 1.5 Flash

Gets:

✍️ Final answer

🤖 Self-reported Gemini confidence score (0–100)

Computes:

📊 Cosine similarity between question and top chunk

🧠 Accuracy Scores
Score	Meaning
📊 Similarity Score	Measures how semantically close your question is to the best matching document chunk (via cosine similarity)
🤖 Gemini Confidence	Gemini's own confidence in the answer it generated using the given context (0–100)

ℹ️ Click the "How it's calculated" dropdown under each metric for explanation.

🧪 Example Usage
Upload SampleVideo.mp4
Ask: "Who won the match?"

✅ App transcribes video, chunks and indexes it

🤖 Gemini returns: "England won the 3rd Test match against India by 22 runs."

📊 Similarity: 84.32%

🤖 Gemini Confidence: 95%

🧰 Tech Stack
Component	Tech Used
🧠 LLM	Google Gemini 1.5 Flash (via API)
📉 Embeddings	sentence-transformers
🔍 Vector search	FAISS
🧾 PDF extraction	PyMuPDF (fitz)
🗣️ Transcription	OpenAI Whisper
📺 YouTube download	yt_dlp
🧪 UI Framework	Streamlit
🔐 Auth	Manual with bcrypt + JSON

📦 Requirements
requirements.txt

txt
Copy
Edit
streamlit
bcrypt
whisper
PyMuPDF
faiss-cpu
sentence-transformers
scikit-learn
yt_dlp
python-dotenv
google-generativeai
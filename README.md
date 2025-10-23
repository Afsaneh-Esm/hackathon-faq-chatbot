#  Hackathon FAQ Chatbot

A simple **multilingual chatbot** built with **FastAPI** and **Streamlit**, designed to answer questions about **tech jobs**, **events**, and **German language courses** in Berlin.

This is a **minimal and modular version** for hackathon development — ready to integrate into a larger platform or website.

---

##  Run the project locally

### 1️⃣ Create a virtual environment
bash
python -m venv .venv
### 2️⃣ Activate the environment
Windows (PowerShell):

bash
Copy code
.\.venv\Scripts\Activate.ps1
Mac/Linux:

bash
Copy code
source .venv/bin/activate
### 3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
### 4️⃣ Run the FastAPI backend
bash
Copy code
python -m uvicorn app.app:app --reload --port 8000
After running, open this link in your browser:
 http://127.0.0.1:8000/docs

🌐 Optional: Streamlit Frontend (UI)
You can also run the visual chatbot interface built in Streamlit with map-based results:

bash
Copy code
streamlit run streamlit_app.py
This will launch the web app locally on:
 http://localhost:8501

📁 Project structure
pgsql
Copy code
hackathon-faq-chatbot/
│

├── app/

│   ├── app.py              ← FastAPI backend (core chatbot logic)

│   ├── data/               ← CSV data files (jobs, events, language courses with lat/lon)

├── streamlit_app.py        ← Streamlit frontend (UI & map)

├── requirements.txt        ← Dependencies

├── requirements_api.txt        ← Dependencies

├── runtime.txt             ← Python version for Streamlit Cloud

└── README.md               ← Full setup & usage guide

⚙️ Dependencies
Python 3.10+

FastAPI

Uvicorn

Pandas

Scikit-learn

Langdetect

Googletrans

Streamlit (for UI)

PyDeck (map rendering)

All dependencies are listed in requirements.txt and requirements_api.txt

 Key Features
Multilingual input detection (English, German, Persian, Arabic, Turkish, Urdu)

Category-specific semantic search using TF-IDF

Integrated map visualization (PyDeck)

Modular design (FastAPI backend + Streamlit frontend)

Clean, easily extensible CSV-based data system

 Live Demo
If deployed on Streamlit Cloud:
👉 https://berlin-tech-bot.streamlit.app



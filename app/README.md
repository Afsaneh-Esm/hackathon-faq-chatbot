Hackathon FAQ Chatbot

A simple chatbot built with FastAPI that answers questions about tech jobs, events, and German language upskilling.
This is a minimal version for hackathon development.

🚀 Run the project locally

1️⃣ Create a virtual environment
python -m venv .venv

2️⃣ Activate the environment

Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

Mac/Linux:
source .venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the FastAPI server
python -m uvicorn app.app:app --reload --port 8000

After running, open this link in your browser:
http://127.0.0.1:8000/docs

Here you can test all API endpoints easily.

💬 Quick test from the terminal

curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"message":"react job berlin"}'

📁 Project structure

hackathon-faq-chatbot/
├── app
│ └── app.py → Main backend code (FastAPI)
├── requirements.txt → Python dependencies
├── .gitignore → Files ignored by Git
└── README.md → Project guide

⚙️ Dependencies

Python 3.10+

FastAPI

Uvicorn

Pandas

Scikit-learn


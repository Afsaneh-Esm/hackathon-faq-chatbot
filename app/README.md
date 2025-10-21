# Hackathon FAQ Chatbot

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.app:app --reload --port 8000

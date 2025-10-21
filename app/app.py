# app/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Hackathon FAQ Chatbot", version="0.1.0")

# --- Test data (keeping it simple for now) ---
data = [
    {"id":"job-001","category":"jobs","title":"Junior Frontend Developer (React) – Berlin","body":"React + TypeScript, hybrid.","url":"https://example.com"},
    {"id":"evt-001","category":"events","title":"Berlin JS Monthly Meetup","body":"Meetup Free for networking.","url":"https://example.com"},
    {"id":"lang-001","category":"language","title":"A2 German Evening Class – VHS Berlin","body":"Low-cost class for immigrants.","url":"https://example.com"},
    {"id":"faq-001","category":"faq","title":"What does the chatbot cover?","body":"Jobs, Events, German upskilling.","url":""},
]
df = pd.DataFrame(data)
df["search_text"] = (df["title"] + " " + df["body"]).str.lower()

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["search_text"].tolist())

class ChatIn(BaseModel):
    message: str
    top_k: int = 3

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/chat")
def chat(body: ChatIn):
    qv = vectorizer.transform([body.message.lower()])
    sims = cosine_similarity(qv, X).flatten()
    idxs = sims.argsort()[::-1][:body.top_k]
    results: List[Dict[str, Any]] = []
    for i in idxs:
        row = df.iloc[int(i)]
        results.append({
            "id": row["id"],
            "title": row["title"],
            "snippet": row["body"],
            "url": row["url"],
            "category": row["category"],
            "score": float(sims[int(i)])
        })
    answer = "Results found." if results else "Nothing found; write your sentence simpler."
    return {"answer": answer, "results": results}

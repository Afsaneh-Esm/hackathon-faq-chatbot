from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

app = FastAPI(title="Hackathon FAQ Chatbot", version="0.1.0")

# ============================================================
#                LOAD REAL DATA FROM CSV FILES
# ============================================================
DATA_DIR = Path("data")

def safe_load(path):
    """Read CSV safely and fill missing values"""
    try:
        df = pd.read_csv(path).fillna("")
        print(f"✅ Loaded {len(df)} rows from {path.name}")
        return df
    except Exception as e:
        print(f"⚠️ Could not read {path.name}: {e}")
        return pd.DataFrame()

# --- JOBS DATA ---
jobs_path = next(DATA_DIR.glob("*job*"), None)
if jobs_path:
    jobs = safe_load(jobs_path)
    jobs["id"] = jobs.get("id", range(len(jobs)))
    jobs["category"] = "jobs"
    jobs["title"] = jobs.get("title", "")
    jobs["body"] = (
        "Company: " + jobs.get("company", "").astype(str)
        + " | Industry: " + jobs.get("company_industry", "").astype(str)
        + " | Location: " + jobs.get("location", "").astype(str)
        + " | Type: " + jobs.get("job_type", "").astype(str)
    )
    jobs["url"] = jobs.get("job_url_direct", jobs.get("job_url", ""))
else:
    jobs = pd.DataFrame()

# --- EVENTS DATA ---
events_path = next(DATA_DIR.glob("*event*"), None)
if events_path:
    events = safe_load(events_path)
    events["id"] = range(len(events))
    events["category"] = "events"
    events["title"] = events.get("Title", "")
    events["body"] = (
        "When: " + events.get("Date & Time", "").astype(str)
        + " | Location: " + events.get("Location", "").astype(str)
    )
    events["url"] = events.get("Link", "")
else:
    events = pd.DataFrame()

# --- LANGUAGE COURSES DATA ---
lang_path = next(DATA_DIR.glob("*german*"), None)
if lang_path:
    lang = safe_load(lang_path)
    lang["id"] = range(len(lang))
    lang["category"] = "language"
    lang["title"] = lang.get("course_name", "")
    lang["body"] = (
        "Provider: " + lang.get("provider", "").astype(str)
        + " | Level: " + lang.get("german_level", "").astype(str)
        + " | Duration: " + lang.get("duration", "").astype(str)
        + " | Price: " + lang.get("price", "").astype(str)
        + " | Location: " + lang.get("location", "").astype(str)
    )
    lang["url"] = lang.get("url", "")
else:
    lang = pd.DataFrame()

# --- MERGE ALL ---
df = pd.concat([jobs, events, lang], ignore_index=True).fillna("")

# fallback in case of no data
if df.empty:
    df = pd.DataFrame([
        {"id": 1, "category": "faq", "title": "What does the chatbot cover?",
         "body": "Jobs, Events, and German language courses in Berlin.", "url": ""}
    ])

# create a searchable column
df["search_text"] = (df["title"].astype(str) + " " + df["body"].astype(str)).str.lower()

print(f"📊 Total records loaded: {len(df)}")

# ============================================================
#                    BUILD SEARCH MODEL
# ============================================================
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["search_text"])

def search_query(query: str, top_k: int = 5):
    # detect intent (category)
    q_lower = query.lower()
    category = None
    if "job" in q_lower or "work" in q_lower or "career" in q_lower:
        category = "jobs"
    elif "event" in q_lower or "meetup" in q_lower or "conference" in q_lower:
        category = "events"
    elif "course" in q_lower or "class" in q_lower or "german" in q_lower:
        category = "language"

    # filter data by detected category
    df_filtered = df
    if category:
        df_filtered = df[df["category"] == category]

    # if nothing found, fallback to full data
    if df_filtered.empty:
        df_filtered = df

    # vectorize and search
    query_vec = vectorizer.transform([query.lower()])
    X_filtered = vectorizer.transform(df_filtered["search_text"])
    scores = cosine_similarity(query_vec, X_filtered)[0]

    top_indices = scores.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        row = df_filtered.iloc[idx]
        results.append({
            "id": row.get("id", ""),
            "category": row.get("category", ""),
            "title": row.get("title", ""),
            "snippet": row.get("body", "")[:200],
            "url": row.get("url", ""),
            "score": float(scores[idx])
        })

    if not results:
        return {"answer": "Sorry, I couldn’t find relevant info."}
    else:
        cat_info = f" in {category}" if category else ""
        return {"answer": f"Found {len(results)} results{cat_info}.", "results": results}


# ============================================================
#                      API ENDPOINTS
# ============================================================

@app.get("/healthz")
def health_check():
    return {"ok": True, "records_loaded": len(df)}

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
def chat(query: ChatInput):
    results = search_query(query.message)
    if not results:
        return {"answer": "Sorry, I couldn’t find relevant info."}
    return {"answer": f"Found {len(results)} results.", "results": results}

# Optional root
@app.get("/")
def root():
    return {"message": "Hackathon FAQ Chatbot is running. Visit /docs for the API."}

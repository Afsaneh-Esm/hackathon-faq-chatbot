# app/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import os
import pandas as pd

# Search / ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# CORS (so a frontend on another origin can call the API)
from fastapi.middleware.cors import CORSMiddleware

# Language detection + translation (with fallback)
from langdetect import detect
try:
    from googletrans import Translator
    _translator = Translator()
except Exception:
    _translator = None

app = FastAPI(title="Hackathon FAQ Chatbot", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # limit to your frontend domain later if needed
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
#               MAP HELPERS
# ==========================================
BERLIN_BOUNDS = {
    "lat_min": 52.2, "lat_max": 52.7,
    "lon_min": 13.0, "lon_max": 13.8,
}
BERLIN_CENTER = (52.5200, 13.4050)

def _as_float(v):
    try:
        return float(v)
    except Exception:
        return None

def _first_present(cols: list[str], df: pd.DataFrame) -> str | None:
    for c in cols:
        if c in df.columns:
            return c
    return None

def _attach_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect many possible latitude/longitude column names, standardize to 'lat' and 'lon',
    and coerce to floats (handling comma decimals too).
    """
    if df is None or df.empty:
        return df
    df = df.copy()

    lat_candidates = ["lat","latitude","lat.","gps_lat","y","Lat","Latitude","LAT","LATITUDE"]
    lon_candidates = ["lon","lng","longitude","long","lon.","gps_lon","x","Lon","Lng","Longitude","LONG","LONGITUDE"]

    cols_lower = {c.lower(): c for c in df.columns}

    def pick(cands):
        for c in cands:
            if c.lower() in cols_lower:
                return cols_lower[c.lower()]
        return None

    lat_col = pick(lat_candidates)
    lon_col = pick(lon_candidates)

    if lat_col and "lat" not in df.columns:
        df["lat"] = df[lat_col]
    if lon_col and "lon" not in df.columns:
        df["lon"] = df[lon_col]

    # Ensure columns exist
    if "lat" not in df.columns:
        df["lat"] = None
    if "lon" not in df.columns:
        df["lon"] = None

    # Coerce to numeric (support comma decimals)
    df["lat"] = pd.to_numeric(df["lat"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    return df

def _within_berlin(lat: float, lon: float) -> bool:
    return (
        BERLIN_BOUNDS["lat_min"] <= lat <= BERLIN_BOUNDS["lat_max"] and
        BERLIN_BOUNDS["lon_min"] <= lon <= BERLIN_BOUNDS["lon_max"]
    )

def build_markers(rows: pd.DataFrame, limit: int = 20):
    """
    Create map markers; if lat/lon invalid or out-of-bounds, fall back to Berlin center
    so the frontend always receives usable markers.
    """
    markers = []
    if rows is None or rows.empty:
        return markers

    for _, r in rows.head(limit).iterrows():
        lat = _as_float(r.get("lat"))
        lon = _as_float(r.get("lon"))

        # Fallback if missing/invalid
        if lat is None or lon is None:
            lat, lon = BERLIN_CENTER

        # Clamp to center if out-of-bounds
        if not _within_berlin(lat, lon):
            lat, lon = BERLIN_CENTER

        markers.append({
            "lat": lat,
            "lon": lon,
            "title": r.get("title",""),
            "url": r.get("url",""),
            "category": r.get("category",""),
        })
    return markers

# ==========================================
#             DATA LOADING (CSV)
# ==========================================

# Resolve DATA_DIR flexibly:
# 1) DATA_DIR env, 2) app/data, 3) app/_data, 4) repo-root/data
_here = Path(__file__).resolve().parent
_candidates = []
env_dir = os.getenv("DATA_DIR")
if env_dir:
    _candidates.append(Path(env_dir))
_candidates += [_here / "data", _here / "_data", _here.parent / "data"]
DATA_DIR = next((p for p in _candidates if p.exists()), _here / "data")

def safe_load(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path).fillna("")
        print(f"✅ Loaded {len(df)} rows from {path.name}")
        return df
    except Exception as e:
        print(f"⚠️ Could not read {path.name}: {e}")
        return pd.DataFrame()

def col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a string Series for a column, or empty strings if missing."""
    if name in df.columns:
        return df[name].astype(str)
    return pd.Series([""] * len(df))

# --- JOBS ---
jobs_path = next((p for p in DATA_DIR.glob("*job*") if p.is_file()), None)
if jobs_path:
    jobs = safe_load(jobs_path)
    if "id" in jobs.columns:
        jobs["id"] = jobs["id"].astype(str)
    else:
        jobs["id"] = [f"job-{i:06d}" for i in range(len(jobs))]
    jobs["category"] = "jobs"
    jobs["title"] = col(jobs, "title")
    # prefer direct URL if present
    j_direct = col(jobs, "job_url_direct")
    j_url = col(jobs, "job_url")
    jobs["url"] = j_direct.where(j_direct.str.len() > 0, j_url)
    jobs["body"] = (
        "Company: " + col(jobs, "company") +
        " | Industry: " + col(jobs, "company_industry") +
        " | Location: " + col(jobs, "location") +
        " | Type: " + col(jobs, "job_type")
    )
    jobs = _attach_latlon(jobs)
else:
    jobs = pd.DataFrame()

# --- EVENTS ---
events_path = next((p for p in DATA_DIR.glob("*event*") if p.is_file()), None)
if events_path:
    events = safe_load(events_path)
    events["id"] = [f"evt-{i:06d}" for i in range(len(events))]
    events["category"] = "events"
    # beware of capitalization in CSV
    title_col = "Title" if "Title" in events.columns else "title"
    date_col = "Date & Time" if "Date & Time" in events.columns else ("date_time" if "date_time" in events.columns else "date")
    loc_col = "Location" if "Location" in events.columns else "location"
    link_col = "Link" if "Link" in events.columns else "link"
    events["title"]  = col(events, title_col)
    events["url"]    = col(events, link_col)
    events["body"]   = "When: " + col(events, date_col) + " | Location: " + col(events, loc_col)
    events = _attach_latlon(events)
else:
    events = pd.DataFrame()

# --- LANGUAGE COURSES ---
lang_path = next((p for p in DATA_DIR.glob("*german*") if p.is_file()), None)
if lang_path:
    lang = safe_load(lang_path)
    lang["id"] = [f"lang-{i:06d}" for i in range(len(lang))]
    lang["category"] = "language"
    lang["title"] = col(lang, "course_name")
    lang["url"]   = col(lang, "url")
    lang["body"]  = (
        "Provider: " + col(lang, "provider") +
        " | Level: " + col(lang, "german_level") +
        " | Duration: " + col(lang, "duration") +
        " | Price: " + col(lang, "price") +
        " | Location: " + col(lang, "location")
    )
    lang = _attach_latlon(lang)
else:
    lang = pd.DataFrame()

# Merge all
df = pd.concat([jobs, events, lang], ignore_index=True).fillna("")
if df.empty:
    df = pd.DataFrame([
        {"id": "sample-1", "category": "faq", "title": "What does the chatbot cover?",
         "url": "", "body": "Jobs, tech events, and German language courses in Berlin.",
         "lat": BERLIN_CENTER[0], "lon": BERLIN_CENTER[1]}
    ])

# --- normalize lat/lon to numeric & keep them inside Berlin bounds ---
if "lat" in df.columns and "lon" in df.columns:
    df["lat"] = pd.to_numeric(df["lat"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    # fallback: if NaN -> center
    df["lat"] = df["lat"].fillna(BERLIN_CENTER[0])
    df["lon"] = df["lon"].fillna(BERLIN_CENTER[1])

    # if out of bounds -> center
    lat_ok = (df["lat"] >= BERLIN_BOUNDS["lat_min"]) & (df["lat"] <= BERLIN_BOUNDS["lat_max"])
    lon_ok = (df["lon"] >= BERLIN_BOUNDS["lon_min"]) & (df["lon"] <= BERLIN_BOUNDS["lon_max"])
    df.loc[~(lat_ok & lon_ok), ["lat", "lon"]] = BERLIN_CENTER

# Build search text
df["search_text"] = (df.get("title", "").astype(str) + " " +
                     df.get("body", "").astype(str)).str.lower()

print(f"📊 Total records loaded: {len(df)}")

# ==========================================
#            SEARCH MODEL (TF-IDF)
# ==========================================
vectorizer = TfidfVectorizer(stop_words="english")
X_full = vectorizer.fit_transform(df["search_text"])

# Simple keyword-based intent
def detect_intent(text: str) -> str | None:
    t = text.lower()
    if any(k in t for k in ["job", "work", "career", "developer", "engineer"]):
        return "jobs"
    if any(k in t for k in ["event", "meetup", "conference", "summit"]):
        return "events"
    if any(k in t for k in ["course", "class", "german", "sprachkurs"]):
        return "language"
    return None

# Language fallback maps (very small, extend as needed)
FALLBACK_MAPS = {
    # German
    "de": {
        "arbeit": "job", "stelle": "job", "entwickler": "developer",
        "veranstaltung": "event", "konferenz": "conference", "meetup": "event",
        "kurs": "course", "deutsch": "german", "sprachkurs": "course",
        "berlin": "berlin", "november": "november", "oktober": "october",
        "frontend": "frontend", "daten": "data", "ki": "ai"
    },
    # Persian
    "fa": {
        "کار": "job", "شغل": "job", "فرانت": "frontend", "بک": "backend",
        "رویداد": "event", "کنفرانس": "conference", "مییتاپ": "event",
        "دوره": "course", "کلاس": "course", "آلمانی": "german",
        "برلین": "berlin"
    },
    # Arabic
    "ar": {
        "وظيفة": "job", "عمل": "job", "مطور": "developer", "مبرمج": "developer",
        "حدث": "event", "فعالية": "event", "مؤتمر": "conference", "ميتاب": "event",
        "دورة": "course", "كورس": "course", "صف": "class", "ألمانية": "german",
        "برلين": "berlin", "ذكاء اصطناعي": "ai", "بيانات": "data", "واجهة": "frontend", "خلفية": "backend"
    },
    # Turkish
    "tr": {
        "iş": "job", "meslek": "job", "geliştirici": "developer", "yazılımcı": "developer",
        "etkinlik": "event", "konferанс": "conference", "buluşma": "event", "meetup": "event",
        "kurs": "course", "ders": "class", "almanca": "german",
        "berlin": "berlin", "yapay zeka": "ai", "veri": "data", "frontend": "frontend", "backend": "backend"
    },
    # Urdu (Arabic script)
    "ur": {
        "نوکری": "job", "ملازمت": "job", "ڈیویلپر": "developer", "انجینئر": "engineer",
        "ایونٹ": "event", "تقریب": "event", "کانفرنس": "conference", "میٹ اپ": "event",
        "کورس": "course", "کلاس": "class", "جرمن": "german",
        "برلن": "berlin", "اے آئی": "ai", "ڈیٹا": "data", "فرنٹ اینڈ": "frontend", "بیک اینڈ": "backend"
    },
}

def translate_to_en(text: str) -> tuple[str, str]:
    """Return (english_text, detected_lang). Uses googletrans if available, else keyword fallback."""
    try:
        lang = detect(text)
    except Exception:
        lang = "en"

    if lang == "en":
        return text, lang

    if _translator:
        try:
            translated = _translator.translate(text, dest="en")
            return translated.text, lang
        except Exception:
            pass  # fall back below

    lowered = text.lower()
    mapping = FALLBACK_MAPS.get(lang, {})
    for k, v in mapping.items():
        lowered = lowered.replace(k, v)
    return lowered, lang

def search_query(query: str, top_k: int = 5):
    # Intent detection (to narrow results)
    category = detect_intent(query)
    df_filtered = df if category is None else df[df["category"] == category]
    if df_filtered.empty:
        df_filtered = df

    # Vectorize filtered set with the same vocabulary
    qv = vectorizer.transform([query.lower()])
    Xf = vectorizer.transform(df_filtered["search_text"])
    scores = cosine_similarity(qv, Xf)[0]

    # Pick top-k
    top_idx = scores.argsort()[-top_k:][::-1]
    top_rows = df_filtered.iloc[top_idx] if len(df_filtered) else df_filtered

    results = []
    for i in top_idx:
        row = df_filtered.iloc[i]
        results.append({
            "id": row.get("id", ""),
            "category": row.get("category", ""),
            "title": row.get("title", ""),
            "snippet": row.get("body", "")[:200],
            "url": row.get("url", ""),
            # expose lat/lon (always numeric & inside Berlin due to normalization)
            "lat": row.get("lat", None),
            "lon": row.get("lon", None),
            "score": float(scores[i])
        })

    # Build map markers from the same top rows
    markers = build_markers(top_rows, limit=top_k)

    answer = f"Found {len(results)} results"
    if category:
        answer += f" in {category}"
    return {"answer": answer + ".", "results": results, "markers": markers}

# ==========================================
#                 API
# ==========================================
@app.get("/healthz")
def healthz():
    return {"ok": True, "records_loaded": int(len(df))}

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
def chat(body: ChatInput):
    # Detect & translate before search
    msg_en, lang = translate_to_en(body.message)
    payload = search_query(msg_en)
    if lang != "en":
        payload["answer"] += f" (detected: {lang} → translated)"
    return payload

@app.get("/")
def root():
    return {"message": "Hackathon FAQ Chatbot is running. Visit /docs"}

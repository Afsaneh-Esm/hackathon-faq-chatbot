# streamlit_app.py
import os
from pathlib import Path
import pandas as pd
import streamlit as st
import pydeck as pdk

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Lang
from langdetect import detect
try:
    from googletrans import Translator
    _translator = Translator()
except Exception:
    _translator = None

# ==========================
# Config
# ==========================
st.set_page_config(page_title="Kiez Connect Chatbot (Streamlit)", layout="wide")

BERLIN_BOUNDS = {"lat_min": 52.2, "lat_max": 52.7, "lon_min": 13.0, "lon_max": 13.8}
BERLIN_CENTER = (52.5200, 13.4050)

# DATA_DIR priority: env → app/data → app/_data → repo-root/data
_here = Path(__file__).resolve().parent
candidates = []
if os.getenv("DATA_DIR"):
    candidates.append(Path(os.getenv("DATA_DIR")))
candidates += [_here / "app" / "data", _here / "app" / "_data", _here / "data"]
DATA_DIR = next((p for p in candidates if p.exists()), _here / "app" / "data")

st.sidebar.markdown("**Data dir:** " + str(DATA_DIR))

def safe_load(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path).fillna("")
        return df
    except Exception:
        # try common encodings
        for enc in ["utf-8-sig", "latin-1"]:
            try:
                return pd.read_csv(path, encoding=enc).fillna("")
            except Exception:
                pass
    return pd.DataFrame()

def col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name].astype(str) if name in df.columns else pd.Series([""] * len(df))

def _attach_latlon(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = df.copy()
    lat_candidates = ["lat","latitude","lat.","gps_lat","y","Lat","Latitude","LAT","LATITUDE"]
    lon_candidates = ["lon","lng","longitude","long","lon.","gps_lon","x","Lon","Lng","Longitude","LONG","LONGITUDE"]
    cols_lower = {c.lower(): c for c in df.columns}
    def pick(cands):
        for c in cands:
            if c.lower() in cols_lower: return cols_lower[c.lower()]
        return None
    lat_col = pick(lat_candidates)
    lon_col = pick(lon_candidates)
    if lat_col and "lat" not in df.columns: df["lat"] = df[lat_col]
    if lon_col and "lon" not in df.columns: df["lon"] = df[lon_col]
    if "lat" not in df.columns: df["lat"] = None
    if "lon" not in df.columns: df["lon"] = None
    # numeric with comma support
    df["lat"] = pd.to_numeric(df["lat"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    # fill/clip to Berlin
    df["lat"] = df["lat"].fillna(BERLIN_CENTER[0])
    df["lon"] = df["lon"].fillna(BERLIN_CENTER[1])
    lat_ok = (df["lat"] >= BERLIN_BOUNDS["lat_min"]) & (df["lat"] <= BERLIN_BOUNDS["lat_max"])
    lon_ok = (df["lon"] >= BERLIN_BOUNDS["lon_min"]) & (df["lon"] <= BERLIN_BOUNDS["lon_max"])
    df.loc[~(lat_ok & lon_ok), ["lat", "lon"]] = BERLIN_CENTER
    return df

# --------------------------
# Load CSVs
# --------------------------
def load_all():
    # jobs
    jobs_path = next((p for p in DATA_DIR.glob("*job*") if p.is_file()), None)
    if jobs_path is not None:
        jobs = safe_load(jobs_path)
        jobs["id"] = jobs["id"].astype(str) if "id" in jobs.columns else [f"job-{i:06d}" for i in range(len(jobs))]
        jobs["category"] = "jobs"
        jobs["title"] = col(jobs, "title")
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

    # events
    events_path = next((p for p in DATA_DIR.glob("*event*") if p.is_file()), None)
    if events_path is not None:
        events = safe_load(events_path)
        events["id"] = [f"evt-{i:06d}" for i in range(len(events))]
        events["category"] = "events"
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

    # language
    lang_path = next((p for p in DATA_DIR.glob("*german*") if p.is_file()), None)
    if lang_path is not None:
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

    df_all = pd.concat([jobs, events, lang], ignore_index=True).fillna("")
    if df_all.empty:
        df_all = pd.DataFrame([{
            "id": "sample-1", "category": "faq", "title": "What does the chatbot cover?",
            "url": "", "body": "Jobs, tech events, and German language courses in Berlin.",
            "lat": BERLIN_CENTER[0], "lon": BERLIN_CENTER[1],
        }])
    return df_all

@st.cache_data(show_spinner=False)
def build_index(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform((df.get("title", "").astype(str) + " " + df.get("body", "").astype(str)).str.lower())
    return vectorizer, X

def detect_intent(text: str):
    t = text.lower()
    if any(k in t for k in ["job", "work", "career", "developer", "engineer"]): return "jobs"
    if any(k in t for k in ["event", "meetup", "conference", "summit"]): return "events"
    if any(k in t for k in ["course", "class", "german", "sprachkurs"]): return "language"
    return None

FALLBACK_MAPS = {
    "de": {"arbeit":"job","stelle":"job","entwickler":"developer","veranstaltung":"event","konferenz":"conference",
           "meetup":"event","kurs":"course","deutsch":"german","sprachkurs":"course","berlin":"berlin"},
    "fa": {"کار":"job","شغل":"job","رویداد":"event","مییتاپ":"event","دوره":"course","کلاس":"course","آلمانی":"german","برلین":"berlin"},
    "ar": {"وظيفة":"job","عمل":"job","حدث":"event","فعالية":"event","مؤتمر":"conference","دورة":"course","ألمانية":"german","برلين":"berlin"},
    "tr": {"iş":"job","etkinlik":"event","konferans":"conference","kurs":"course","almanca":"german","berlin":"berlin"},
    "ur": {"نوکری":"job","ایونٹ":"event","کورس":"course","جرمن":"german","برلن":"berlin"},
}

def translate_to_en(text: str):
    try:
        lang = detect(text)
    except Exception:
        lang = "en"
    if lang == "en": return text, lang
    if _translator:
        try:
            return _translator.translate(text, dest="en").text, lang
        except Exception:
            pass
    lowered = text.lower()
    mapping = FALLBACK_MAPS.get(lang, {})
    for k, v in mapping.items():
        lowered = lowered.replace(k, v)
    return lowered, lang

def search(df: pd.DataFrame, vectorizer, X, q: str, top_k=5):
    cat = detect_intent(q)
    df_f = df if cat is None else df[df["category"] == cat]
    if df_f.empty: df_f = df
    qv = vectorizer.transform([q.lower()])
    Xf = vectorizer.transform((df_f.get("title","").astype(str) + " " + df_f.get("body","").astype(str)).str.lower())
    scores = cosine_similarity(qv, Xf)[0]
    idx = scores.argsort()[-top_k:][::-1]
    rows = df_f.iloc[idx]
    results = []
    for i in idx:
        r = df_f.iloc[i]
        results.append({
            "id": r.get("id",""), "category": r.get("category",""), "title": r.get("title",""),
            "snippet": str(r.get("body",""))[:200], "url": r.get("url",""),
            "lat": float(r.get("lat", BERLIN_CENTER[0])), "lon": float(r.get("lon", BERLIN_CENTER[1])),
            "score": float(scores[i])
        })
    return results, rows

# ==========================
# UI
# ==========================
st.title("Kiez Connect – Streamlit Chat & Map")
st.caption("Search jobs, tech events, and German courses in Berlin. Multilingual input supported.")

df = load_all()
vectorizer, X = build_index(df)

with st.sidebar:
    st.subheader("Search")
    user_msg = st.text_input("Ask me anything", value="events in Kreuzberg")
    k = st.slider("Top K", 3, 20, 5)
    st.markdown("---")
    st.write("Loaded records:", len(df))
    st.write("Categories:", ", ".join(sorted(df["category"].unique())))

if user_msg:
    q_en, lang = translate_to_en(user_msg)
    results, rows = search(df, vectorizer, X, q_en, top_k=k)

    # Map
    st.subheader("Map")
    map_df = rows.copy()
    map_df = map_df[["title","lat","lon","category","url"]].rename(columns={"lon":"lng"})
    # Pydeck scatter
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[lng, lat]',
        get_radius=60,
        pickable=True,
    )
    view_state = pdk.ViewState(latitude=BERLIN_CENTER[0], longitude=BERLIN_CENTER[1], zoom=11)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{title}\n{category}"}))

    # Results list
    st.subheader("Results")
    for r in results:
        st.markdown(f"**{r['title']}**  \n*{r['category']}*  —  [link]({r['url']})")
        st.caption(f"lat: {r['lat']:.5f}, lon: {r['lon']:.5f} | score: {r['score']:.3f}")
        st.write(r["snippet"])
        st.markdown("---")

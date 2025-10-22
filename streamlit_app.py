# streamlit_app.py
import os
from pathlib import Path
import pandas as pd
import streamlit as st
import pydeck as pdk
from langdetect import detect
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="Kiez Connect (Lite)", layout="wide")

BERLIN_BOUNDS = {"lat_min": 52.2, "lat_max": 52.7, "lon_min": 13.0, "lon_max": 13.8}
BERLIN_CENTER = (52.5200, 13.4050)

# DATA_DIR: env → app/data → app/_data → data
_here = Path(__file__).resolve().parent
candidates = [Path(os.getenv("DATA_DIR"))] if os.getenv("DATA_DIR") else []
candidates += [_here / "app" / "data", _here / "app" / "_data", _here / "data"]
DATA_DIR = next((p for p in candidates if p.exists()), _here / "app" / "data")

def safe_load(p: Path) -> pd.DataFrame:
    for enc in (None, "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc).fillna("")
        except Exception:
            pass
    return pd.DataFrame()

def col(df, name): return df[name].astype(str) if name in df.columns else pd.Series([""]*len(df))

def _attach_latlon(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = df.copy()
    lat_candidates = ["lat","latitude","lat.","gps_lat","y","Lat","Latitude"]
    lon_candidates = ["lon","lng","longitude","long","lon.","gps_lon","x","Lon","Lng","Longitude"]
    low = {c.lower(): c for c in df.columns}
    def pick(cands):
        for c in cands:
            if c.lower() in low: return low[c.lower()]
        return None
    lat_col, lon_col = pick(lat_candidates), pick(lon_candidates)
    if lat_col and "lat" not in df.columns: df["lat"] = df[lat_col]
    if lon_col and "lon" not in df.columns: df["lon"] = df[lon_col]
    if "lat" not in df.columns: df["lat"] = None
    if "lon" not in df.columns: df["lon"] = None
    df["lat"] = pd.to_numeric(df["lat"].astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(BERLIN_CENTER[0])
    df["lon"] = pd.to_numeric(df["lon"].astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(BERLIN_CENTER[1])
    lat_ok = (df["lat"] >= BERLIN_BOUNDS["lat_min"]) & (df["lat"] <= BERLIN_BOUNDS["lat_max"])
    lon_ok = (df["lon"] >= BERLIN_BOUNDS["lon_min"]) & (df["lon"] <= BERLIN_BOUNDS["lon_max"])
    df.loc[~(lat_ok & lon_ok), ["lat","lon"]] = BERLIN_CENTER
    return df

def load_all():
    # jobs
    jp = next((p for p in DATA_DIR.glob("*job*") if p.is_file()), None)
    if jp:
        jobs = safe_load(jp)
        jobs["id"] = jobs["id"].astype(str) if "id" in jobs.columns else [f"job-{i:06d}" for i in range(len(jobs))]
        jobs["category"] = "jobs"
        jobs["title"] = col(jobs, "title")
        j_direct, j_url = col(jobs, "job_url_direct"), col(jobs, "job_url")
        jobs["url"] = j_direct.where(j_direct.str.len() > 0, j_url)
        jobs["body"] = ("Company: " + col(jobs,"company") + " | Industry: " + col(jobs,"company_industry") +
                        " | Location: " + col(jobs,"location") + " | Type: " + col(jobs,"job_type"))
        jobs = _attach_latlon(jobs)
    else:
        jobs = pd.DataFrame()

    # events
    ep = next((p for p in DATA_DIR.glob("*event*") if p.is_file()), None)
    if ep:
        events = safe_load(ep)
        events["id"] = [f"evt-{i:06d}" for i in range(len(events))]
        events["category"] = "events"
        title_col = "Title" if "Title" in events.columns else "title"
        date_col = "Date & Time" if "Date & Time" in events.columns else ("date_time" if "date_time" in events.columns else "date")
        loc_col = "Location" if "Location" in events.columns else "location"
        link_col = "Link" if "Link" in events.columns else "link"
        events["title"] = col(events, title_col)
        events["url"]   = col(events, link_col)
        events["body"]  = "When: " + col(events, date_col) + " | Location: " + col(events, loc_col)
        events = _attach_latlon(events)
    else:
        events = pd.DataFrame()

    # language
    lp = next((p for p in DATA_DIR.glob("*german*") if p.is_file()), None)
    if lp:
        lang = safe_load(lp)
        lang["id"] = [f"lang-{i:06d}" for i in range(len(lang))]
        lang["category"] = "language"
        lang["title"] = col(lang, "course_name")
        lang["url"]   = col(lang, "url")
        lang["body"]  = ("Provider: " + col(lang,"provider") + " | Level: " + col(lang,"german_level") +
                         " | Duration: " + col(lang,"duration") + " | Price: " + col(lang,"price") +
                         " | Location: " + col(lang,"location"))
        lang = _attach_latlon(lang)
    else:
        lang = pd.DataFrame()

    df = pd.concat([jobs, events, lang], ignore_index=True).fillna("")
    if df.empty:
        df = pd.DataFrame([{"id":"sample-1","category":"faq","title":"What does the chatbot cover?",
                            "url":"","body":"Jobs, tech events, and German language courses in Berlin.",
                            "lat":BERLIN_CENTER[0],"lon":BERLIN_CENTER[1]}])
    df["search_text"] = (df.get("title","").astype(str) + " " + df.get("body","").astype(str)).str.lower()
    return df

@st.cache_data(show_spinner=False)
def build_bm25(corpus):
    tokenized = [doc.split() for doc in corpus]
    return BM25Okapi(tokenized), tokenized

def detect_intent(t: str):
    t = t.lower()
    if any(k in t for k in ["job","work","career","developer","engineer"]): return "jobs"
    if any(k in t for k in ["event","meetup","conference","summit"]): return "events"
    if any(k in t for k in ["course","class","german","sprachkurs"]): return "language"
    return None

FALLBACK_MAPS = {
    "de":{"arbeit":"job","stelle":"job","entwickler":"developer","veranstaltung":"event","konferenz":"conference","kurs":"course","deutsch":"german","sprachkurs":"course","berlin":"berlin"},
    "fa":{"کار":"job","شغل":"job","رویداد":"event","دوره":"course","کلاس":"course","آلمانی":"german","برلین":"berlin"},
    "ar":{"وظيفة":"job","عمل":"job","حدث":"event","فعالية":"event","مؤتمر":"conference","دورة":"course","ألمانية":"german","برلين":"berlin"},
    "tr":{"iş":"job","etkinlik":"event","konferans":"conference","kurs":"course","almanca":"german","berlin":"berlin"},
    "ur":{"نوکری":"job","ایونٹ":"event","کورس":"course","جرمن":"german","برلن":"berlin"},
}

def to_en(text: str):
    try:
        lang = detect(text)
    except Exception:
        lang = "en"
    if lang != "en":
        lowered = text.lower()
        mapping = FALLBACK_MAPS.get(lang, {})
        for k, v in mapping.items(): lowered = lowered.replace(k, v)
        return lowered, lang
    return text, lang

# ============== UI ==============
st.title("Kiez Connect – Streamlit (Lite)")
df = load_all()
bm25, tokenized = build_bm25(df["search_text"].tolist())

with st.sidebar:
    st.write("Data dir:", str(DATA_DIR))
    q = st.text_input("Ask me anything", "events in Kreuzberg")
    k = st.slider("Top K", 3, 20, 5)
    st.write("Loaded:", len(df))
    st.write("Categories:", ", ".join(sorted(df["category"].unique())))

if q:
    q_en, lang = to_en(q)
    # category filter
    cat = detect_intent(q_en)
    df_f = df if cat is None else df[df["category"] == cat]
    if df_f.empty: df_f = df
    # rebuild BM25 for filtered
    bm25_f, toks_f = build_bm25(df_f["search_text"].tolist())
    scores = bm25_f.get_scores(q_en.split())
    idx = list(pd.Series(scores).nlargest(k).index)
    rows = df_f.iloc[idx]

    st.subheader("Map")
    map_df = rows[["title","lat","lon","category","url"]].rename(columns={"lon":"lng"}).copy()
    layer = pdk.Layer("ScatterplotLayer", data=map_df, get_position='[lng, lat]', get_radius=60, pickable=True)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(latitude=BERLIN_CENTER[0], longitude=BERLIN_CENTER[1], zoom=11), tooltip={"text":"{title}\n{category}"}))

    st.subheader("Results")
    for _, r in rows.iterrows():
        st.markdown(f"**{r.get('title','')}**  \n*{r.get('category','')}* — [link]({r.get('url','')})")
        st.caption(f"lat: {float(r['lat']):.5f}, lon: {float(r['lon']):.5f}")
        st.write(str(r.get("body",""))[:240])
        st.markdown("---")


"""
app.py  -  Customer Emotion Analyzer (Streamlit web app).

Run with:  streamlit run app.py

Give it customer reviews (sample file, your own CSV, or pasted text).
An AI model tags each review with an emotion, then the app shows:
  - how customers feel overall
  - what the angry / disappointed / delighted customers are saying
"""

import pandas as pd
import streamlit as st

from emotions import EMOTION_INFO, NEGATIVE, analyze, load_model, summary

st.set_page_config(page_title="Customer Emotion Analyzer", page_icon="💬", layout="wide")


@st.cache_resource(show_spinner="Loading the AI model (first time downloads ~330 MB)...")
def get_model():
    return load_model()


st.title("💬 Customer Emotion Analyzer")
st.write("Customers don't just like or dislike a product. They feel **angry, disappointed, "
         "delighted, worried...** This app uses AI to read reviews and find those emotions.")

# ---- Try one sentence ----
with st.expander("✏️ Try a single review first", expanded=False):
    one = st.text_input("Type a review", "The delivery was late again and nobody answered my calls!")
    if st.button("Analyze this review"):
        r = analyze(get_model(), [one]).iloc[0]
        st.subheader(f"{r['Customer mood']} ({r['Confidence %']}% confident)")
        st.caption(EMOTION_INFO[r["Emotion"]][2])

# ---- Choose reviews ----
st.header("1. Choose your reviews")
source = st.radio("Where are the reviews?",
                  ["Use the sample reviews", "Upload a CSV file", "Paste reviews"], horizontal=True)

texts = []
if source == "Use the sample reviews":
    sample = pd.read_csv("sample_reviews.csv")
    st.caption(f"{len(sample)} sample reviews of a wireless earphone (written for this demo).")
    texts = sample["review"].tolist()
elif source == "Upload a CSV file":
    up = st.file_uploader("CSV file with one review per row", type="csv")
    if up:
        data = pd.read_csv(up)
        column = st.selectbox("Which column has the review text?", data.columns)
        texts = data[column].dropna().astype(str).tolist()
        st.caption(f"{len(texts)} reviews found.")
else:
    pasted = st.text_area("Paste reviews, one per line", height=180)
    texts = [line for line in pasted.splitlines() if line.strip()]

# ---- Analyze ----
st.header("2. Analyze")
if st.button("🔍 Find the emotions", type="primary", disabled=not texts):
    with st.spinner(f"Reading {len(texts)} reviews..."):
        st.session_state["results"] = analyze(get_model(), texts)

if "results" not in st.session_state:
    st.stop()

results = st.session_state["results"]
s = summary(results)

# ---- Results ----
st.header("3. Results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews analyzed", s["total"])
c2.metric("😊 Delighted", f"{s['happy']:.0f}%")
c3.metric("😠 Unhappy (angry, disappointed, disgusted, worried)", f"{s['unhappy']:.0f}%")
c4.metric("Most common mood", EMOTION_INFO[s["top"]][1])

left, right = st.columns([1, 1])
with left:
    st.subheader("How customers feel")
    chart = s["share"].rename(index=lambda e: f"{EMOTION_INFO[e][0]} {EMOTION_INFO[e][1]}")
    st.bar_chart(chart)
with right:
    st.subheader("What each mood means for the business")
    for emo in s["share"].index:
        emoji, meaning, action = EMOTION_INFO[emo]
        st.markdown(f"**{emoji} {meaning}** ({s['share'][emo]:.0f}%): {action}")

st.subheader("Read the reviews by mood")
moods = [EMOTION_INFO[e][1] for e in s["share"].index]
default = [EMOTION_INFO[e][1] for e in NEGATIVE if e in s["share"].index]
pick = st.multiselect("Show moods", moods, default=default or moods)
keep = [e for e, info in EMOTION_INFO.items() if info[1] in pick]
view = results[results["Emotion"].isin(keep)].sort_values("Confidence %", ascending=False)
st.dataframe(view[["Customer mood", "Confidence %", "Review"]], use_container_width=True, hide_index=True)

st.download_button("Download all results as CSV", results.to_csv(index=False), "review_emotions.csv")
st.caption("AI model: j-hartmann/emotion-english-distilroberta-base (Hugging Face). "
           "Works best on English text. Like any AI, it can misread sarcasm.")

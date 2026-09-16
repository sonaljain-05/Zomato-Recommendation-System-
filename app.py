import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Zomato AI", page_icon="🍽️", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
.main {background: #fff7f8;}
h1 {color:#b1124b;}
.stButton>button {
    width:100%; border-radius:12px; height:3em;
    background:#b1124b; color:white; font-weight:bold;
}
.card {
    padding:20px; border-radius:18px; margin:12px 0;
    background:white; box-shadow:0 4px 15px #ddd;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD ----------
df = pd.read_pickle("restaurant_data_small.pkl")

z = np.load("tfidf_matrix.npz", allow_pickle=False)
tfidf = csr_matrix(
    (z["data"], z["indices"], z["indptr"]),
    shape=tuple(z["shape"])
)

# ---------- HEADER ----------
st.title("🍽️ Zomato AI Recommender")
st.caption("✨ Discover restaurants similar to your choice")

names = sorted(df["name"].dropna().unique())
selected = st.selectbox("🔎 Choose a restaurant", names)

# ---------- RECOMMEND ----------
if st.button("🚀 Find Similar Restaurants"):

    idx = df[df["name"].str.lower() == selected.lower()].index[0]

    scores = cosine_similarity(tfidf[idx], tfidf)[0]
    top = scores.argsort()[-11:][::-1]
    top = [i for i in top if i != idx][:10]

    st.subheader("💡 Recommended For You")

    for i in top:
        r = df.iloc[i]

        st.markdown(f"""
        <div class="card">
        <h3>🍴 {r['name']}</h3>
        📍 {r['location']}<br>
        🍛 {r['cuisines']}<br>
        ⭐ Rating: {r['rate']} &nbsp;&nbsp;
        👍 Votes: {r['votes']}<br>
        📊 Similarity: {scores[i]:.3f}
        </div>
        """, unsafe_allow_html=True)

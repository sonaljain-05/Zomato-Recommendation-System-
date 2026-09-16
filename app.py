
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Zomato AI", page_icon="🍽️", layout="wide")

# ---------- DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#08090d,#170b18,#0b1220);
    color:white;
}
.hero {
    padding:35px;
    border-radius:25px;
    background:linear-gradient(120deg,#ff1768,#7c35ff);
    margin-bottom:25px;
}
.hero h1 {font-size:45px;color:white;margin:0;}
.hero p {font-size:18px;color:#eee;}

.card {
    background:rgba(255,255,255,.08);
    border:1px solid rgba(255,255,255,.15);
    border-radius:22px;
    padding:14px;
    margin-bottom:20px;
    box-shadow:0 8px 30px #0008;
}

.card img {
    width:100%;
    height:190px;
    object-fit:cover;
    border-radius:16px;
}

.badge {
    background:#ff1768;
    padding:5px 12px;
    border-radius:20px;
}

.stButton>button {
    width:100%;
    height:50px;
    border:0;
    border-radius:14px;
    background:linear-gradient(90deg,#ff1768,#8b35ff);
    color:white;
    font-weight:bold;
    font-size:17px;
}
</style>
""", unsafe_allow_html=True)


# ---------- DATA ----------
df = pd.read_pickle("restaurant_data_small.pkl")

z = np.load("tfidf_matrix.npz", allow_pickle=False)

tfidf = csr_matrix(
    (z["data"], z["indices"], z["indptr"]),
    shape=tuple(z["shape"])
)


# ---------- HERO ----------
st.markdown("""
<div class="hero">
<h1>🍽️ Zomato AI</h1>
<p>Discover delicious restaurants using AI-powered recommendations</p>
</div>
""", unsafe_allow_html=True)


# ---------- STATS ----------
c1, c2, c3 = st.columns(3)

c1.metric("🍴 Restaurants", f"{len(df):,}")
c2.metric("⭐ Average Rating", f"{df['rate'].mean():.1f}")
c3.metric("🧠 AI Features", f"{tfidf.shape[1]:,}")


# ---------- SEARCH ----------
st.write("")
names = sorted(df["name"].dropna().unique())

selected = st.selectbox(
    "🔎 Choose a restaurant",
    names
)


# ---------- RECOMMEND ----------
if st.button("✨ Generate Recommendations"):

    idx = df[
        df["name"].str.lower() == selected.lower()
    ].index[0]

    scores = cosine_similarity(
        tfidf[idx],
        tfidf
    )[0]

    top = scores.argsort()[-11:][::-1]
    top = [i for i in top if i != idx][:10]

    st.markdown("## 🍴 Recommended Restaurants")

    # Images for different food categories
    images = [
        "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38",
        "https://images.unsplash.com/photo-1563379926898-05f4575a45d8",
        "https://images.unsplash.com/photo-1547592180-85f173990554",
        "https://images.unsplash.com/photo-1555939594-58d7cb561ad1",
        "https://images.unsplash.com/photo-1513104890138-7c749659a591"
    ]

    for start in range(0, len(top), 2):

        cols = st.columns(2)

        for col, i in zip(cols, top[start:start+2]):

            r = df.iloc[i]
            image = images[start % len(images)]

            with col:
                st.markdown(f"""
                <div class="card">
                    <img src="{image}">
                    <h2>🍴 {r['name']}</h2>
                    <span class="badge">
                        AI Match {scores[i]:.1%}
                    </span>
                    <p>📍 {r['location']}</p>
                    <p>🍛 {r['cuisines']}</p>
                    <p>
                        ⭐ {r['rate']}
                        &nbsp;&nbsp; 👍 {r['votes']:,}
                    </p>
                </div>
                """, unsafe_allow_html=True

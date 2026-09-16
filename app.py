
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Zomato AI",
    page_icon="🍽️",
    layout="wide"
)

# ---------- STYLE ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#08090d,#170b18,#0b1220);
    color: white;
}
.hero {
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(120deg,#ff1768,#7c35ff);
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 45px;
    color: white;
    margin: 0;
}
.hero p {
    font-size: 18px;
    color: #eee;
}
.card {
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.15);
    border-radius: 22px;
    padding: 14px;
    margin-bottom: 20px;
    box-shadow: 0 8px 30px #0008;
}
.card img {
    width: 100%;
    height: 190px;
    object-fit: cover;
    border-radius: 16px;
}
.badge {
    background: #ff1768;
    padding: 5px 12px;
    border-radius: 20px;
}
.stButton > button {
    width: 100%;
    height: 50px;
    border: 0;
    border-radius: 14px;
    background: linear-gradient(90deg,#ff1768,#8b35ff);
    color: white;
    font-weight: bold;
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD DATA ----------
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


# ---------- SELECT RESTAURANT ----------
st.write("")

names = sorted(df["n]()



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

# ---------- CSS ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b0b12, #1b0d1c, #101525);
    color: white;
}
.hero {
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(120deg, #ff1768, #7c35ff);
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 42px;
}
.card {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 15px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.15);
}
.card img {
    width: 100%;
    height: 180px;
    object-fit: cover;
    border-radius: 15px;
}
.badge {
    background: #ff1768;
    padding: 6px 12px;
    border-radius: 20px;
}
.stButton button {
    width: 100%;
    background: linear-gradient(90deg,#ff1768,#7c35ff);
    color: white;
    border: none;
    border-radius: 12px;
    height: 50px;
    font-weight: bold;
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
st.markdown("""
<div class="hero">
    <h1>🍽️ Zomato AI</h1>
    <p>Smart Restaurant Recommendation System</p>
</div>
""", unsafe_allow_html=True)

# ---------- STATS ----------
c1, c2, c3 = st.columns(3)

c1.metric("🍴 Restaurants", len(df))



import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Zomato Recommendation",
    page_icon="🍴",
    layout="centered"
)

# ---------- BACKGROUND ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fff5f5, #fff0e6);
}

.block-container {
    max-width: 850px;
    padding-top: 35px;
}

.title-box {
    text-align: center;
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.food-img {
    display: block;
    margin: 15px auto 25px auto;
    width: 100%;
    max-width: 650px;
    height: 230px;
    object-fit: cover;
    border-radius: 22px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    font-size: 17px;
    font-weight: bold;
}

.card {
    background: white;
    padding: 15px;
    border-radius: 18px;
    margin: 12px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
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


# ---------- HEADER ----------
st.markdown("""
<div class="title-box">
    <h1>🍴 Zomato Restaurant Finder</h1>
    <p>Discover restaurants similar to your favourite place</p>
</div>
""", unsafe_allow_html=True)


# ---------- FOOD IMAGE ----------
st.markdown("""
<img class="food-img"
src="https://images.unsplash.com/photo-1517248135467-4c7edcad34c4">
""", unsafe_allow_html=True)


# ---------- SELECT ----------
names = sorted(df["name"].dropna().unique())

selected = st.selectbox(
    "🔎 Select Restaurant",
    names
)


# ---------- BUTTON ----------
if st.button(
    "🔍 Find Similar Restaurants",
    type="primary",
    width="stretch"
):

    index = df[
        df["name"].str.lower() == selected.lower()
    ].index[0]

    scores = cosine_similarity(
        tfidf[inde]()

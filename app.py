```python id="k8m3x2"
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="FoodMatch AI",
    page_icon="🍽️",
    layout="centered"
)

# ---------- DATA ----------
df = pd.read_pickle("restaurant_data_small.pkl")

z = np.load("tfidf_matrix.npz", allow_pickle=False)
tfidf = csr_matrix(
    (z["data"], z["indices"], z["indptr"]),
    shape=tuple(z["shape"])
)

# ---------- FRONT DESIGN ----------
st.markdown("""
<style>

.block-container {
    max-width: 850px;
    padding-top: 2rem;
}

.hero {
    text-align: center;
    padding: 10px 0 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    color: #777;
    font-size: 17px;
}

.search-box {
    background: #f7f7f7;
    padding: 22px;
    border-radius: 20px;
    margin: 10px 0 25px;
}

.restaurant {
    background: white;
    border-radius: 20px;
    padding: 12px;
    margin: 10px 0;
    box-shadow: 0 4px 18px rgba(0,0,0,.10);
}

.restaurant img {
    width: 100%;
    height: 170px;
    object-fit: cover;
    border-radius: 15px;
}

.match {
    display: inline-block;
    background: #e8f8ef;
    color: #16834b;
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 13px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <h1>🍽️ FoodMatch AI</h1>
    <p>Discover your next favourite restaurant</p>
</div>
""", unsafe_allow_html=True)


# ---------- MAIN IMAGE ----------
st.im
```

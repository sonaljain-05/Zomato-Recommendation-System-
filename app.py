import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------

# PAGE CONFIG

# --------------------------------------------------

st.set_page_config(
page_title="Zomato Recommendation System",
page_icon="🍽️",
layout="centered"
)

# --------------------------------------------------

# DARK UI

# --------------------------------------------------

st.markdown("""

<style>

.stApp {
    background-color: #18181b;
    color: white;
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
}

h1 {
    text-align: center;
    color: white;
}

.subtitle {
    text-align: center;
    color: #bdbdbd;
    margin-bottom: 25px;
}

.hero {
    display: flex;
    justify-content: center;
    margin-bottom: 25px;
}

.hero img {
    width: 100%;
    max-width: 700px;
    height: 260px;
    object-fit: cover;
    border-radius: 18px;
}

.restaurant-card {
    background-color: #27272a;
    padding: 18px;
    border-radius: 15px;
    margin-top: 15px;
    border: 1px solid #3f3f46;
}

.restaurant-name {
    font-size: 20px;
    font-weight: bold;
    color: white;
}

.info {
    color: #d4d4d8;
    margin-top: 5px;
}

.score {
    color: #fbbf24;
    font-weight: bold;
}

</style>

""", unsafe_allow_html=True)

# --------------------------------------------------

# TITLE

# --------------------------------------------------

st.title("🍽️ Zomato Recommendation System")

st.markdown(
'<p class="subtitle">Find restaurants similar to your favourite restaurant</p>',
unsafe_allow_html=True
)

# --------------------------------------------------

# HERO IMAGE

# --------------------------------------------------

st.markdown("""

<div class="hero">
    <img src="https://images.unsplash.com/photo-1517248135467-4c7edcad34c4">
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------

# LOAD RESTAURANT DATA

# --------------------------------------------------

data_file = "restaurant_data_small.pkl"
matrix_file = "tfidf_matrix.npz"

if not os.path.exists(data_file):
st.error("restaurant_data_small.pkl file not found.")
st.stop()

if not os.path.exists(matrix_file):
st.error("tfidf_matrix.npz file not found.")
st.stop()

df = pd.read_pickle(data_file)

# --------------------------------------------------

# CHECK COLUMNS

# --------------------------------------------------

required_columns = [
"name",
"location",
"cuisines",
"rate",
"votes"
]

missing_columns = [
col for col in required_columns
if col not in df.columns
]

if missing_columns:
st.error(
"Required columns are missing: "
+ ", ".join(missing_columns)
)
st.stop()

# --------------------------------------------------

# LOAD TF-IDF MATRIX

# ------------------------------------

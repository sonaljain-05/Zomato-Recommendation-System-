import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
page_title="Zomato Recommendation System",
page_icon="🍽️",
layout="centered"
)

st.markdown(
""" <style>


.stApp {
    background: #fff7ed;
    color: #27272a;
}

.block-container {
    max-width: 900px;
    padding-top: 25px;
    padding-bottom: 50px;
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    color: #7f1d1d;
    margin-bottom: 4px;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #71717a;
    margin-bottom: 25px;
}

.hero {
    width: 100%;
    height: 330px;
    object-fit: cover;
    border-radius: 20px;
    margin-bottom: 25px;
}

.section-title {
    color: #7f1d1d;
    font-size: 24px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 15px;
}

.restaurant-card {
    background: #ffffff;
    border-radius: 18px;
    padding: 20px;
    margin-top: 15px;
    border: 1px solid #fed7aa;
    box-shadow: 0px 4px 14px rgba(0, 0, 0, 0.08);
}

.restaurant-name {
    color: #7f1d1d;
    font-size: 21px;
    font-weight: 750;
    margin-bottom: 10px;
}

.restaurant-info {
    color: #52525b;
    font-size: 14px;
    line-height: 1.7;
}

.similarity {
    color: #b45309;
    font-size: 15px;
    font-weight: 700;
    margin-top: 8px;
}

</style>
""",
unsafe_allow_html=True


)

st.markdown(
'<div class="main-title">🍽️ Zomato Recommendation System</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">Discover restaurants similar to your favourite place</div>',
unsafe_allow_html=True
)

st.markdown(
""" <img
     class="hero"
     src="https://images.unsplash.com/photo-1517248135467-4c7edcad34c4"
     alt="Restaurant"
 >
""",
unsafe_allow_html=True
)

df = pd.read_pickle("restaurant_data_small.pkl")

z = np.load(
"tfidf_matrix.npz",
allow_pickle=False
)

tfidf_matrix = csr_matrix(
(
z["data"],
z["indices"],
z["indptr"]
),
shape=tuple(z["shape"])
)

df = df.reset_index(drop=True)

restaurant_names = sorted(
df["name"].dropna().astype(str).unique()
)

selected_restaurant = st.selectbox(
"🍴 Select a restaurant",
restaurant_names
)

selected_index = df.index[
df["name"].astype(str) == selected]

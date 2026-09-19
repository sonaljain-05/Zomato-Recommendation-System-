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
background-color: #18181b;
color: white;
}


.block-container {
    max-width: 850px;
    padding-top: 30px;
}

.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #bdbdbd;
    margin-bottom: 25px;
}

.card {
    background-color: #27272a;
    padding: 18px;
    margin-top: 15px;
    border-radius: 15px;
    border: 1px solid #3f3f46;
}

.name {
    font-size: 21px;
    font-weight: bold;
    color: white;
}

.details {
    color: #d4d4d8;
    margin-top: 6px;
}

.similarity {
    color: #fbbf24;
    font-weight: bold;
    margin-top: 8px;
}
</style>
""",
unsafe_allow_html=True


)

st.markdown(
'<div class="title">🍽️ Zomato Recommendation System</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">Find restaurants similar to your favourite restaurant</div>',
unsafe_allow_html=True
)

st.image(
"https://images.unsplash.com/photo-1517248135467-4c7edcad34c4",
use_container_width=True
)

# Load restaurant data

df = pd.read_pickle("restaurant_data_small.pkl")

# Load TF-IDF matrix manually

z = np.load("tfidf_matrix.npz", allow_pickle=False)

data = z["data"]
indices = z["indices"]
indptr = z["indptr"]
shape = tuple(z["shape"])

tfidf_matrix = csr_matrix(
(data, indices, indptr),
shape=shape
)

# Restaurant names

restaurant_names = sorted(
df["name"].dropna().astype(str).unique()
)

selected = st.selectbox(
"🍴 Select a restaurant",
restaurant_names
)

# Recommendation button

if st.button(
"🔍 Find Similar Restaurants",
type="primary",
use_container_width=True
):


selected_rows = df[
    df["name"].astype(str).str.lower()
    == selected.lower()
]

index = selected_rows.index[0]

similarity = cosine_similarity(
    tfidf_matrix[index],
    tfidf_matrix
)[0]

best_indexes = similarity.argsort()[-11:][::-1]

st.subheader("✨ Recommended Restaurants")

count = 0

for i in best_indexes:

    if i == index:
        continue

    row = df.iloc[i]

    st.markdown(


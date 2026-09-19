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
    font-size: 36px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #bdbdbd;
    margin-bottom: 25px;
}

.box {
    background-color: #27272a;
    padding: 18px;
    border-radius: 15px;
    margin-top: 15px;
    border: 1px solid #3f3f46;
}

.restaurant {
    font-size: 20px;
    font-weight: bold;
    color: white;
}

.text {
    color: #d4d4d8;
    margin-top: 6px;
}

.score {
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
df["name"].astype(str) == selected_restaurant
][0]

similarity_scores = cosine_similarity(
tfidf_matrix[selected_index],
tfidf_matrix
).flatten()

result = df.copy()

result["Similarity Score"] = similarity_scores

result = result.drop(
index=selected_index
)

result = result.sort_values(
"Similarity Score",
ascending=False
)

result = result.head(10)

result["Rating"] = result["rate"].round(1)

result["Votes"] = result["votes"].astype(int)

result["Similarity"] = (
result["Similarity Score"]
.round(2)
)

result = result[
[
"name",
"location",
"cuisines",
"Rating",
"Votes",
"Similarity"
]
]

st.subheader("✨ Recommended Restaurants")

st.dataframe(
result,
use_container_width=True,)

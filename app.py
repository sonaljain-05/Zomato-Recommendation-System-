import streamlit as st
import pandas as pd
import numpy as np
import os
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
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #bdbdbd;
    font-size: 16px;
    margin-bottom: 25px;
}

.hero-box {
    background-color: #27272a;
    border-radius: 18px;
    padding: 10px;
    margin-bottom: 25px;
}

.hero-box img {
    width: 100%;
    height: 250px;
    object-fit: cover;
    border-radius: 14px;
}

.restaurant-card {
    background-color: #27272a;
    border: 1px solid #3f3f46;
    border-radius: 15px;
    padding: 18px;
    margin-top: 15px;
    margin-bottom: 15px;
}

.restaurant-name {
    font-size: 21px;
    font-weight: 700;
    color: white;
    margin-bottom: 8px;
}

.restaurant-info {
    color: #d4d4d8;
    font-size: 14px;
    line-height: 1.8;
}

.score {
    color: #fbbf24;
    font-weight: 600;
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
'<div class="subtitle">Find restaurants similar to your choice</div>',
unsafe_allow_html=True
)

st.markdown(
""" <div class="hero-box"> <img
         src="https://images.unsplash.com/photo-1517248135467-4c7edcad34c4"
         alt="Restaurant"
     > </div>
""",
unsafe_allow_html=True
)

@st.cache_data
def load_restaurant_data():
file_path = "restaurant_data_small.pkl"


if not os.path.exists(file_path):
    return None

return pd.read_pickle(file_path)


@st.cache_resource
def load_tfidf_matrix():
file_path = "tfidf_matrix.npz"


if not os.path.exists(file_path):
    return None

z = np.load(
    file_path,
    allow_pickle=False
)

data = z["data"]
indices = z["indices"]
indptr = z["indptr"]
shape = tuple(z["shape"])

return csr_matrix(
    (data, indices, indptr),
    shape=shape
)


df = load_restaurant_data()
tfidf_matrix = load_tfidf_matrix()

if df is None:
st.error("restaurant_data_small.pkl file nahi mili.")
st.info(
"restaurant_data_small.pkl ko app.py ke same folder mein upload karo."
)
st.stop()

if tfidf_matrix is None:
st.error("tfidf_matrix.npz file nahi mili.")
st.info(
"tfidf_matrix.npz ko app.py ke same folder mein upload karo."
)
st.stop()

df = df.reset_index(drop=True)

required_columns = [
"name",
"location",
"cuisines",
"rate",
"votes"
]

missing_columns = [
column
for column in required_columns
if column not in df.columns
]

if missing_columns:
st.error("Dataset mein required columns missing hain.")
st.write("Missing columns:", missing_columns)
st.write("Available columns:", list(df.columns))
st.stop()

if len(df) != tfidf_matrix.shape[0]:
st.error(
"Dataset aur TF-IDF matrix ka size match nahi kar raha."
)


st.write("Dataset rows:", len(df))
st.write("TF-IDF rows:", tfidf_matrix.shape[0])

st.stop()


names = sorted(
df["name"]
.dropna()
.astype(str)
.unique()
)

st.markdown("### 🔎 Select a Restaurant")

selected_restaurant = st.selectbox(
"Choose restaurant",
names
)

def recommend(name):


matches = df[
    df["name"]
    .astype(str)
    .str.lower()
    == name.lower()
]

if len(matches) == 0:
    return pd.DataFrame()

index_position = matches.index[0]

scores = cosine_similarity(
    tfidf_matrix[index_position],
    tfidf_matrix
).flatten()

top_indices = scores.argsort()[::-1]

top_indices = [
    i
    for i in top_indices
    if i != index_position
][:10]

result = df.iloc[top_indices].copy()

result["Similarity Score"] = scores[top_indices]

return result[
    [
        "name",
        "location",
        "cuisines",
        "rate",
        "votes",
        "Similarity Score"
    ]
]


find_button = st.button(
"🔍 Find Similar Restaurants",
use_container_width=True
)

if find_button:


recommendations = recommend(
    selected_restaurant
)

if recommendations.empty:

    st.warning("No similar restaurants found.")

else:

    st.markdown("### 🍴 Recommended Restaurants")

    for _, row in recommendations.iterrows():

        restaurant_name = str(row["name"])
        location = str(row["location"])
        cuisines = str(row["cuisines"])

        try:
            rating = float(row["rate"])
        except (ValueError, TypeError):
            rating = 0.0

        try:
            votes = int(float(row["votes"]))
        except (ValueError, TypeError):
            votes = 0

        try:
            similarity = float(
                row["Similarity Score"]
            )
        except (ValueError, TypeError):
            similarity = 0.0

        st.markdown(
            f"""
            <div class="restaurant-card">

                <div class="restaurant-name">
                    🍴 {restaurant_name}
                </div>

                <div class="restaurant-info">

                    📍 <b>Location:</b>
                    {location}
                    <br>

                    🍛 <b>Cuisines:</b>
                    {cuisines}
                    <br>

                    ⭐ <b>Rating:</b>
                    {rating:.1f}
                    <br>

                    👍 <b>Votes:</b>
                    {votes:,}
                    <br>

                    <span class="score">
                        🎯 <b>Similarity Score:</b>
                        {similarity:.2f}
                    </span>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


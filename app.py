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

st.title("🍽️ Zomato Recommendation System")

st.markdown(
'<p class="subtitle">Find restaurants similar to your favourite restaurant</p>',
unsafe_allow_html=True
)

st.markdown("""

<div class="hero">
    <img src="https://images.unsplash.com/photo-1517248135467-4c7edcad34c4">
</div>
""", unsafe_allow_html=True)

data_file = "restaurant_data_small.pkl"
matrix_file = "tfidf_matrix.npz"

if not os.path.exists(data_file):
st.error("restaurant_data_small.pkl file not found.")
st.stop()

if not os.path.exists(matrix_file):
st.error("tfidf_matrix.npz file not found.")
st.stop()

df = pd.read_pickle(data_file)

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

try:
z = np.load(
matrix_file,
allow_pickle=False
)

```
data = z["data"]
indices = z["indices"]
indptr = z["indptr"]
shape = tuple(z["shape"])

tfidf_matrix = csr_matrix(
    (data, indices, indptr),
    shape=shape
)
```

except Exception as e:
st.error("TF-IDF matrix load nahi ho paayi.")
st.write(e)
st.stop()

restaurant_names = sorted(
df["name"]
.dropna()
.astype(str)
.unique()
)

selected_restaurant = st.selectbox(
"🍴 Select a restaurant",
restaurant_names
)

def recommend(name):

```
matches = df[
    df["name"].astype(str).str.lower()
    == name.lower()
]

if len(matches) == 0:
    return pd.DataFrame()

index = matches.index[0]

scores = cosine_similarity(
    tfidf_matrix[index],
    tfidf_matrix
)[0]

top_indices = scores.argsort()[-11:][::-1]

top_indices = [
    i
    for i in top_indices
    if i != index
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
```

if st.button(
"🔍 Find Similar Restaurants",
use_container_width=True,
type="primary"
):

```
recommendations = recommend(
    selected_restaurant
)

if recommendations.empty:
    st.warning("No recommendations found.")

else:
    st.subheader("✨ Recommended Restaurants")

    for _, row in recommendations.iterrows():

        rating = row["rate"]
        votes = row["votes"]
        similarity = row["Similarity Score"]

        st.markdown(
            f"""
            <div class="restaurant-card">

                <div class="restaurant-name">
                    🍴 {row["name"]}
                </div>

                <div class="info">
                    📍 {row["location"]}
                </div>

                <div class="info">
                    🍛 {row["cuisines"]}
                </div>

                <div class="info">
                    ⭐ Rating: {rating:.1f}
                </div>

                <div class="info">
                    👍 Votes: {int(votes)}
                </div>

                <div class="score">
                    🎯 Similarity Score: {similarity:.2f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

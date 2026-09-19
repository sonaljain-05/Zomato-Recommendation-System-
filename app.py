import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Zomato Recommendation",
    page_icon="🍽️",
    layout="centered"
)

# CSS
st.markdown("""
<style>
.stApp {
    background-color: #18181b;
    color: white;
}

.block-container {
    max-width: 700px;
    padding-top: 25px;
}

.title {
    text-align: center;
    font-size: 30px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #aaa;
    margin-bottom: 15px;
}

.card {
    background-color: #27272a;
    padding: 14px;
    border-radius: 12px;
    margin: 8px 0;
    border: 1px solid #3f3f46;
}

.name {
    font-size: 18px;
    font-weight: bold;
}

.info {
    color: #ccc;
    font-size: 14px;
    margin-top: 5px;
}

.score {
    color: #fbbf24;
    font-weight: bold;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="title">🍽️ Zomato Recommendation</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Find restaurants similar to your favourite</div>',
    unsafe_allow_html=True
)

# Load data
df = pd.read_pickle("restaurant_data_small.pkl")

z = np.load("tfidf_matrix.npz", allow_pickle=False)

tfidf_matrix = csr_matrix(
    (z["data"], z["indices"], z["indptr"]),
    shape=tuple(z["shape"])
)

df = df.reset_index(drop=True)

restaurant_names = sorted(
    df["name"].dropna().astype(str).unique()
)

# Restaurant selection
selected_restaurant = st.selectbox(
    "🍴 Select a restaurant",
    restaurant_names
)

# Button
if st.button(
    "🔍 Get Recommendations",
    use_container_width=True
):

    selected_index = df.index[
        df["name"].astype(str) == selected_restaurant
    ][0]

    # Similarity
    similarity_scores = cosine_similarity(
        tfidf_matrix[selected_index],
        tfidf_matrix
    ).flatten()

    result = df.copy()

    result["Similarity"] = similarity_scores

    result = result.drop(index=selected_index)

    result = result.sort_values(
        "Similarity",
        ascending=False
    ).head(10)

    st.subheader("✨ Recommended Restaurants")

    # Display as cards instead of table
    for _, row in result.iterrows():

        rating = pd.to_numeric(
            row["rate"],
            errors="coerce"
        )

        votes = pd.to_numeric(
            row["votes"],
            errors="coerce"
        )

        st.markdown(
            f"""
            <div class="card">
                <div class="name">
                    🍴 {row['name']}
                </div>

                <div class="info">
                    📍 {row['location']}
                </div>

                <div class="info">
                    🍜 {row['cuisines']}
                </div>

                <div class="info">
                    ⭐ Rating: {rating:.1f}
                    &nbsp;&nbsp; 👥 Votes: {int(votes)}
                </div>

                <div class="score">
                    Similarity: {row['Similarity']:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.caption(
    "Powered by TF-IDF & Cosine Similarity"
)

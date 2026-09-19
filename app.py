import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Zomato Recommendation",
    page_icon="🍽️",
    layout="centered"
)


# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

.stApp {
    background-color: #18181b;
    color: white;
}

.block-container {
    max-width: 700px;
    padding-top: 20px;
    padding-bottom: 30px;
}

.title {
    text-align: center;
    font-size: 30px;
    font-weight: 700;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 14px;
    color: #a1a1aa;
    margin-bottom: 20px;
}

.card {
    background-color: #27272a;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
    border: 1px solid #3f3f46;
}

.restaurant-name {
    font-size: 18px;
    font-weight: 700;
    color: white;
    margin-bottom: 7px;
}

.info {
    font-size: 14px;
    color: #d4d4d8;
    margin: 4px 0;
}

.score {
    font-size: 14px;
    color: #fbbf24;
    font-weight: 700;
    margin-top: 7px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================
st.markdown(
    '<div class="title">🍽️ Zomato Recommendation</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Find restaurants similar to your favourite restaurant'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# LOAD DATA
# ==========================================
try:
    df = pd.read_pickle("restaurant_data_small.pkl")
except Exception as e:
    st.error("❌ Could not load restaurant_data_small.pkl")
    st.stop()


# ==========================================
# LOAD TF-IDF MATRIX
# ==========================================
try:
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

except Exception as e:
    st.error("❌ Could not load tfidf_matrix.npz")
    st.stop()


# ==========================================
# PREPARE DATA
# ==========================================
df = df.reset_index(drop=True)

df["name"] = df["name"].fillna("Unknown Restaurant").astype(str)

restaurant_names = sorted(
    df["name"].unique()
)


# ==========================================
# RESTAURANT SELECTION
# ==========================================
selected_restaurant = st.selectbox(
    "🍴 Select a restaurant",
    restaurant_names
)


# ==========================================
# RECOMMENDATION BUTTON
# ==========================================
if st.button(
    "🔍 Get Recommendations",
    use_container_width=True
):

    # Find selected restaurant
    selected_indices = df.index[
        df["name"] == selected_restaurant
    ].tolist()

    if len(selected_indices) == 0:
        st.error("Restaurant not found.")
        st.stop()

    selected_index = selected_indices[0]


    # ======================================
    # COSINE SIMILARITY
    # ======================================
    similarity_scores = cosine_similarity(
        tfidf_matrix[selected_index],
        tfidf_matrix
    ).flatten()


    # ======================================
    # CREATE RESULT
    # ======================================
    result = df.copy()

    result["Similarity"] = similarity_scores


    # Remove selected restaurant
    result = result.drop(
        index=selected_index
    )


    # Sort by similarity
    result = result.sort_values(
        "Similarity",
        ascending=False
    )


    # ======================================
    # REMOVE DUPLICATES
    # ======================================
    result = result.drop_duplicates(
        subset=["name", "location"],
        keep="first"
    )


    # Top 10
    result = result.head(10)


    # ======================================
    # DISPLAY
    # ======================================
    st.subheader("✨ Recommended Restaurants")


    for _, row in result.iterrows():

        # Rating
        rating = pd.to_numeric(
            row.get("rate", 0),
            errors="coerce"
        )

        if pd.isna(rating):
            rating_text = "N/A"
        else:
            rating_text = f"{rating:.1f}"


        # Votes
        votes = pd.to_numeric(
            row.get("votes", 0),
            errors="coerce"
        )

        if pd.isna(votes):
            votes = 0


        # Location
        location = row.get(
            "location",
            "Location not available"
        )

        if pd.isna(location):
            location = "Location not available"


        # Cuisines
        cuisines = row.get(
            "cuisines",
            "Not available"
        )

        if pd.isna(cuisines):
            cuisines = "Not available"


        # Similarity
        similarity = float(
            row["Similarity"]
        )


        # ==================================
        # RESTAURANT CARD
        # ==================================
        st.markdown(
            f"""
            <div class="card">

                <div class="restaurant-name">
                    🍴 {row["name"]}
                </div>

                <div class="info">
                    📍 {location}
                </div>

                <div class="info">
                    🍜 {cuisines}
                </div>

                <div class="info">
                    ⭐ Rating: {rating_text}
                    &nbsp;&nbsp; 👥 Votes: {int(votes)}
                </div>

                <div class="score">
                    Similarity: {similarity:.2f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# FOOTER
# ==========================================
st.caption(
    "Recommendations are generated using TF-IDF and Cosine Similarity."
)

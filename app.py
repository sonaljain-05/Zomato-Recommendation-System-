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
# SIMPLE CSS
# ==========================================
st.markdown("""
<style>
.stApp {
    background-color: #18181b;
}

.block-container {
    max-width: 700px;
    padding-top: 20px;
}

h1, h2, h3, p, label {
    color: white !important;
}

.restaurant-card {
    padding: 12px;
    margin-bottom: 10px;
    border-radius: 12px;
    background: #27272a;
    border: 1px solid #3f3f46;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# TITLE
# ==========================================
st.markdown(
    "<h1 style='text-align:center;'>🍽️ Zomato Recommendation</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:#aaa;'>"
    "Find restaurants similar to your favourite restaurant"
    "</p>",
    unsafe_allow_html=True
)


# ==========================================
# LOAD DATA
# ==========================================
try:
    df = pd.read_pickle("restaurant_data_small.pkl")
except FileNotFoundError:
    st.error("restaurant_data_small.pkl file नहीं मिली.")
    st.stop()


# ==========================================
# LOAD TF-IDF
# ==========================================
try:
    z = np.load("tfidf_matrix.npz", allow_pickle=False)

    tfidf_matrix = csr_matrix(
        (
            z["data"],
            z["indices"],
            z["indptr"]
        ),
        shape=tuple(z["shape"])
    )

except FileNotFoundError:
    st.error("tfidf_matrix.npz file नहीं मिली.")
    st.stop()


# ==========================================
# RESET INDEX
# ==========================================
df = df.reset_index(drop=True)

df["name"] = df["name"].fillna(
    "Unknown Restaurant"
).astype(str)

df["location"] = df["location"].fillna(
    "Location not available"
).astype(str)

df["cuisines"] = df["cuisines"].fillna(
    "Not available"
).astype(str)


# ==========================================
# RESTAURANT LIST
# ==========================================
restaurant_names = sorted(
    df["name"].unique()
)


# ==========================================
# SELECT RESTAURANT
# ==========================================
selected_restaurant = st.selectbox(
    "🍴 Select a restaurant",
    restaurant_names
)


# ==========================================
# BUTTON
# ==========================================
recommend_button = st.button(
    "🔍 Get Recommendations",
    use_container_width=True
)


# ==========================================
# RECOMMENDATIONS
# ==========================================
if recommend_button:

    selected_index = df.index[
        df["name"] == selected_restaurant
    ][0]


    # --------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------
    similarity_scores = cosine_similarity(
        tfidf_matrix[selected_index],
        tfidf_matrix
    ).flatten()


    result = df.copy()

    result["Similarity"] = similarity_scores


    # --------------------------------------
    # REMOVE SELECTED RESTAURANT
    # --------------------------------------
    result = result[
        result["name"] != selected_restaurant
    ]


    # --------------------------------------
    # SORT
    # --------------------------------------
    result = result.sort_values(
        "Similarity",
        ascending=False
    )


    # --------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------
    result = result.drop_duplicates(
        subset=["name", "location"],
        keep="first"
    )


    # --------------------------------------
    # TOP 10
    # --------------------------------------
    result = result.head(10)


    # ======================================
    # HEADING
    # ======================================
    st.markdown("### ✨ Recommended Restaurants")


    # ======================================
    # DISPLAY RESTAURANTS
    # ======================================
    for _, row in result.iterrows():

        # Rating
        rating = pd.to_numeric(
            row["rate"],
            errors="coerce"
        )

        if pd.isna(rating):
            rating = "N/A"
        else:
            rating = f"{rating:.1f}"


        # Votes
        votes = pd.to_numeric(
            row["votes"],
            errors="coerce"
        )

        if pd.isna(votes):
            votes = 0

        votes = int(votes)


        # Similarity
        similarity = float(
            row["Similarity"]
        )


        # ----------------------------------
        # NATIVE STREAMLIT CARD
        # ----------------------------------
        with st.container(border=True):

            st.markdown(
                f"### 🍴 {row['name']}"
            )

            st.write(
                f"📍 **Location:** {row['location']}"
            )

            st.write(
                f"🍜 **Cuisine:** {row['cuisines']}"
            )

            st.write(
                f"⭐ **Rating:** {rating}   "
                f"👥 **Votes:** {votes}"
            )

            st.write(
                f"🟡 **Similarity:** {similarity:.2f}"
            )


# ==========================================
# FOOTER
# ==========================================
st.caption(
    "Powered by TF-IDF and Cosine Similarity"
)

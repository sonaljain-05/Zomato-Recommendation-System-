import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import random


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Zomato AI Recommendations",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, #3b1717 0%, #18181b 35%),
        #18181b;
    color: white;
}

.block-container {
    max-width: 900px;
    padding-top: 25px;
    padding-bottom: 50px;
}


/* ---------- Header ---------- */

.hero-title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 4px;
    color: #ffffff;
}

.hero-subtitle {
    text-align: center;
    color: #a1a1aa;
    font-size: 15px;
    margin-bottom: 25px;
}


/* ---------- Selection Box ---------- */

.select-label {
    font-size: 15px;
    font-weight: 600;
    color: #f4f4f5;
    margin-bottom: 7px;
}


/* ---------- Recommendation Button ---------- */

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    background: linear-gradient(90deg, #ef4444, #f97316);
    color: white;
    border: none;
    font-size: 16px;
    font-weight: 600;
    transition: 0.2s;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0px 8px 25px rgba(239, 68, 68, 0.30);
}


/* ---------- Section Heading ---------- */

.section-title {
    font-size: 24px;
    font-weight: 700;
    color: white;
    margin-top: 30px;
    margin-bottom: 15px;
}


/* ---------- Restaurant Card ---------- */

.restaurant-card {
    background: rgba(39, 39, 42, 0.92);
    border: 1px solid #3f3f46;
    border-radius: 18px;
    padding: 12px;
    margin-bottom: 15px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.18);
}

.restaurant-image {
    border-radius: 13px;
    width: 100%;
    height: 190px;
    object-fit: cover;
}

.restaurant-name {
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
    margin-top: 10px;
}

.restaurant-location {
    color: #a1a1aa;
    font-size: 13px;
    margin-top: 3px;
}

.restaurant-cuisine {
    color: #d4d4d8;
    font-size: 14px;
    margin-top: 8px;
}

.rating {
    color: #fbbf24;
    font-weight: 600;
    font-size: 14px;
}

.similarity {
    color: #fb7185;
    font-weight: 600;
    font-size: 14px;
}


/* ---------- Footer ---------- */

.footer {
    text-align: center;
    color: #71717a;
    font-size: 12px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HERO
# =========================================================

st.markdown(
    '<div class="hero-title">🍽️ Zomato AI Recommendations</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'Discover restaurants similar to your favourite places'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# HERO IMAGE
# =========================================================

st.image(
    "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4"
    "?auto=format&fit=crop&w=1200&q=85",
    use_container_width=True
)


# =========================================================
# LOAD DATA
# =========================================================

try:
    df = pd.read_pickle("restaurant_data_small.pkl")

except FileNotFoundError:
    st.error("❌ restaurant_data_small.pkl file नहीं मिली.")
    st.stop()


# =========================================================
# LOAD TF-IDF MATRIX
# =========================================================

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

except FileNotFoundError:

    st.error("❌ tfidf_matrix.npz file नहीं मिली.")
    st.stop()


# =========================================================
# DATA CLEANING
# =========================================================

df = df.reset_index(drop=True)

df["name"] = (
    df["name"]
    .fillna("Unknown Restaurant")
    .astype(str)
)

df["location"] = (
    df["location"]
    .fillna("Location unavailable")
    .astype(str)
)

df["cuisines"] = (
    df["cuisines"]
    .fillna("Various cuisines")
    .astype(str)
)


# =========================================================
# RESTAURANT SELECTOR
# =========================================================

restaurant_names = sorted(
    df["name"].unique()
)

st.markdown(
    '<div class="select-label">🍴 Choose your favourite restaurant</div>',
    unsafe_allow_html=True
)

selected_restaurant = st.selectbox(
    "Restaurant",
    restaurant_names,
    label_visibility="collapsed"
)


# =========================================================
# BUTTON
# =========================================================

recommend = st.button(
    "✨ Find Similar Restaurants"
)


# =========================================================
# FOOD IMAGE FUNCTION
# =========================================================

def get_food_image(cuisine):

    cuisine = str(cuisine).lower()

    if "pizza" in cuisine:
        query = "pizza food restaurant"

    elif "burger" in cuisine:
        query = "burger food restaurant"

    elif "biryani" in cuisine:
        query = "biryani indian food"

    elif "south indian" in cuisine:
        query = "south indian food dosa"

    elif "north indian" in cuisine:
        query = "north indian food"

    elif "chinese" in cuisine:
        query = "chinese food restaurant"

    elif "cafe" in cuisine:
        query = "coffee cafe food"

    elif "dessert" in cuisine:
        query = "dessert food"

    elif "sandwich" in cuisine:
        query = "sandwich food"

    elif "italian" in cuisine:
        query = "italian pasta food"

    else:
        query = "restaurant food"

    # Random photo from Unsplash Source
    return (
        "https://images.unsplash.com/"
        "photo-1504674900247-0877df9cc836"
        "?auto=format&fit=crop&w=900&q=80"
    )


# =========================================================
# RECOMMENDATIONS
# =========================================================

if recommend:

    # Find selected restaurant
    selected_indices = df.index[
        df["name"] == selected_restaurant
    ].tolist()

    if not selected_indices:
        st.error("Restaurant नहीं मिला.")
        st.stop()

    selected_index = selected_indices[0]


    # -----------------------------------------------------
    # COSINE SIMILARITY
    # -----------------------------------------------------

    similarity_scores = cosine_similarity(
        tfidf_matrix[selected_index],
        tfidf_matrix
    ).flatten()


    result = df.copy()

    result["Similarity"] = similarity_scores


    # -----------------------------------------------------
    # REMOVE SELECTED RESTAURANT
    # -----------------------------------------------------

    result = result[
        result["name"] != selected_restaurant
    ]


    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    result = result.sort_values(
        "Similarity",
        ascending=False
    )


    # -----------------------------------------------------
    # REMOVE DUPLICATES
    # -----------------------------------------------------

    result = result.drop_duplicates(
        subset=["name", "location"],
        keep="first"
    )


    # -----------------------------------------------------
    # TOP 8
    # -----------------------------------------------------

    result = result.head(8)


    # =====================================================
    # HEADING
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '✨ Recommended For You'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # RESTAURANT CARDS
    # =====================================================

    for _, row in result.iterrows():

        # Rating
        rating = pd.to_numeric(
            row["rate"],
            errors="coerce"
        )

        if pd.isna(rating):
            rating_text = "N/A"
        else:
            rating_text = f"{rating:.1f}"


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


        # Food image
        food_image = get_food_image(
            row["cuisines"]
        )


        # -------------------------------------------------
        # IMAGE
        # -------------------------------------------------

        st.image(
            food_image,
            use_container_width=True
        )


        # -------------------------------------------------
        # RESTAURANT NAME
        # -------------------------------------------------

        st.markdown(
            f"### 🍴 {row['name']}"
        )


        # -------------------------------------------------
        # LOCATION
        # -------------------------------------------------

        st.caption(
            f"📍 {row['location']}"
        )


        # -------------------------------------------------
        # CUISINE
        # -------------------------------------------------

        st.write(
            f"🍜 {row['cuisines']}"
        )


        # -------------------------------------------------
        # RATING + VOTES + SIMILARITY
        # -------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Rating",
                f"⭐ {rating_text}"
            )

        with col2:
            st.metric(
                "Votes",
                f"👥 {votes}"
            )

        with col3:
            st.metric(
                "Match",
                f"{similarity:.0%}"
            )


        st.divider()


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    'Powered by TF-IDF • Cosine Similarity • Streamlit'
    '</div>',
    unsafe_allow_html=True
)

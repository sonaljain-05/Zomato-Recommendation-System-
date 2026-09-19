
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Zomato AI Recommendation",
    page_icon="🍽️",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ==============================
       MAIN BACKGROUND
       ============================== */

    .stApp {
        background: linear-gradient(
            135deg,
            #fff7ed 0%,
            #ffedd5 50%,
            #fef3c7 100%
        );

        color: #292524;
    }


    /* ==============================
       MAIN CONTAINER
       ============================== */

    .block-container {
        max-width: 1250px;
        padding-top: 45px;
        padding-bottom: 60px;
    }


    /* ==============================
       MAIN TITLE
       ============================== */

    .main-title {
        font-size: 46px;
        font-weight: 800;
        color: #292524;

        margin-top: 5px;
        margin-bottom: 8px;

        letter-spacing: -1px;
    }


    .subtitle {
        color: #78716c;

        font-size: 17px;

        margin-bottom: 40px;
    }


    /* ==============================
       LEFT CARD
       ============================== */

    .left-box {
        background-color: rgba(255, 255, 255, 0.92);

        border: 1px solid #fed7aa;

        border-radius: 24px;

        padding: 24px;

        box-shadow:
            0 12px 35px rgba(120, 53, 15, 0.10);

        min-height: 700px;
    }


    /* ==============================
       LEFT IMAGE
       ============================== */

    .left-image-space {
        margin-top: 5px;
        margin-bottom: 25px;
    }


    /* ==============================
       LEFT HEADING
       ============================== */

    .left-heading {
        font-size: 31px;

        font-weight: 800;

        color: #292524;

        margin-top: 20px;

        margin-bottom: 12px;
    }


    .left-text {
        color: #57534e;

        font-size: 15px;

        line-height: 1.8;

        margin-bottom: 25px;
    }


    /* ==============================
       RIGHT SECTION
       ============================== */

    .right-section {
        padding-left: 15px;
        padding-top: 8px;
    }


    .recommend-title {
        font-size: 32px;

        font-weight: 800;

        color: #292524;

        margin-bottom: 8px;
    }


    .recommend-subtitle {
        color: #78716c;

        font-size: 15px;

        margin-bottom: 28px;
    }


    /* ==============================
       RESTAURANT NAME
       ============================== */

    .restaurant-name {
        font-size: 20px;

        font-weight: 700;

        color: #292524;

        margin-bottom: 7px;
    }


    .restaurant-info {
        color: #57534e;

        font-size: 13px;

        margin: 6px 0;
    }


    .match {
        color: #ea580c;

        font-weight: 800;

        font-size: 14px;

        margin-top: 10px;
    }


    /* ==============================
       BUTTON
       ============================== */

    div.stButton > button {

        width: 100%;

        height: 52px;

        border-radius: 13px;

        border: none;

        background: linear-gradient(
            90deg,
            #f97316,
            #ea580c
        );

        color: white;

        font-size: 16px;

        font-weight: 700;

        margin-top: 8px;
    }


    div.stButton > button:hover {

        background: linear-gradient(
            90deg,
            #ea580c,
            #c2410c
        );

        color: white;

        box-shadow:
            0 8px 20px rgba(234, 88, 12, 0.25);
    }


    /* ==============================
       SELECTBOX
       ============================== */

    div[data-baseweb="select"] > div {

        border-radius: 12px;

        border: 1px solid #fed7aa;

        background-color: white;
    }


    /* ==============================
       IMAGES
       ============================== */

    div[data-testid="stImage"] img {

        border-radius: 16px;

    }


    /* ==============================
       RECOMMENDATION SPACING
       ============================== */

    div[data-testid="stHorizontalBlock"] {

        margin-bottom: 8px;
    }


    /* ==============================
       DIVIDER
       ============================== */

    hr {

        border-color: #fed7aa;

        margin-top: 20px;

        margin-bottom: 22px;
    }


    /* ==============================
       MOBILE
       ============================== */

    @media (max-width: 768px) {

        .main-title {
            font-size: 34px;
        }

        .subtitle {
            font-size: 14px;
            margin-bottom: 25px;
        }

        .left-heading {
            font-size: 26px;
        }

        .recommend-title {
            font-size: 26px;
        }

        .right-section {
            padding-left: 0;
            margin-top: 30px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🍽️ Zomato AI Recommendation'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover restaurants similar to the ones you already love.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD RESTAURANT DATA
# =========================================================

try:

    df = pd.read_pickle(
        "restaurant_data_small.pkl"
    )

except FileNotFoundError:

    st.error(
        "❌ restaurant_data_small.pkl file नहीं मिली।"
    )

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

    st.error(
        "❌ tfidf_matrix.npz file नहीं मिली।"
    )

    st.stop()


# =========================================================
# CLEAN DATA
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
# FIVE FOOD IMAGES
#
# First 5 = different
# Then repeat
# =========================================================

food_images = [

    # 1 - Pizza
    "https://images.unsplash.com/photo-1574071318508-1cdbab80d002?auto=format&fit=crop&w=800&q=85",

    # 2 - Burger
    "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?auto=format&fit=crop&w=800&q=85",

    # 3 - Biryani
    "https://images.unsplash.com/photo-1563379091339-03246963d51a?auto=format&fit=crop&w=800&q=85",

    # 4 - South Indian
    "https://images.unsplash.com/photo-1630383249896-424e482df921?auto=format&fit=crop&w=800&q=85",

    # 5 - Chinese
    "https://images.unsplash.com/photo-1563245372-f21724e3856d?auto=format&fit=crop&w=800&q=85"
]


# =========================================================
# MAIN TWO COLUMN LAYOUT
# =========================================================

left_column, right_column = st.columns(
    [0.95, 1.05],
    gap="large"
)


# =========================================================
# LEFT SIDE
# =========================================================

with left_column:

    st.markdown(
        '<div class="left-box">',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # BIG STATIC FOOD IMAGE
    # -----------------------------------------------------

    st.image(
        "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?auto=format&fit=crop&w=1200&q=95",
        use_container_width=True
    )


    # -----------------------------------------------------
    # LEFT HEADING
    # -----------------------------------------------------

    st.markdown(
        '<div class="left-heading">'
        'Find Your Next Favourite 🍴'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # LEFT DESCRIPTION
    # -----------------------------------------------------

    st.markdown(
        '<div class="left-text">'
        'Select a restaurant you already love and let our '
        'recommendation system discover similar restaurants '
        'for you.'
        '<br><br>'
        '✨ Smart recommendations<br>'
        '🍜 Cuisine similarity<br>'
        '🎯 TF-IDF + Cosine Similarity'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# RIGHT SIDE
# =========================================================

with right_column:

    st.markdown(
        '<div class="right-section">',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # RIGHT TITLE
    # -----------------------------------------------------

    st.markdown(
        '<div class="recommend-title">'
        '🍴 Restaurant Recommendations'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="recommend-subtitle">'
        'Choose a restaurant and discover similar places.'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # RESTAURANT SELECTBOX
    # -----------------------------------------------------

    restaurant_names = sorted(
        df["name"].unique()
    )


    selected_restaurant = st.selectbox(
        "Select Restaurant",
        restaurant_names
    )


    # -----------------------------------------------------
    # BUTTON
    # -----------------------------------------------------

    recommend_button = st.button(
        "✨ Show Recommendations"
    )


    # =====================================================
    # RECOMMENDATION LOGIC
    # =====================================================

    if recommend_button:

        # Find selected restaurant
        selected_index = df.index[
            df["name"] == selected_restaurant
        ][0]


        # -------------------------------------------------
        # COSINE SIMILARITY
        # -------------------------------------------------

        similarity_scores = cosine_similarity(
            tfidf_matrix[selected_index],
            tfidf_matrix
        ).flatten()


        # -------------------------------------------------
        # CREATE RESULT
        # -------------------------------------------------

        result = df.copy()

        result["Similarity"] = similarity_scores


        # Remove selected restaurant
        result = result[
            result["name"] != selected_restaurant
        ]


        # Sort by similarity
        result = result.sort_values(
            "Similarity",
            ascending=False
        )


        # Remove duplicate restaurants
        result = result.drop_duplicates(
            subset=["name", "location"],
            keep="first"
        )


        # Top 10 recommendations
        result = result.head(10)


        # -------------------------------------------------
        # RECOMMENDATION HEADING
        # -------------------------------------------------

        st.markdown(
            "### ✨ Recommended For You"
        )


        # =================================================
        # DISPLAY RECOMMENDATIONS
        # =================================================

        for recommendation_number, (_, row) in enumerate(
            result.iterrows()
        ):


            # -------------------------------------------------
            # IMAGE REPEAT LOGIC
            #
            # 1 -> Image 1
            # 2 -> Image 2
            # 3 -> Image 3
            # 4 -> Image 4
            # 5 -> Image 5
            # 6 -> Image 1
            # 7 -> Image 2
            # ...
            # -------------------------------------------------

            image_index = (
                recommendation_number % 5
            )

            image_url = food_images[
                image_index
            ]


            # -------------------------------------------------
            # RATING
            # -------------------------------------------------

            rating = pd.to_numeric(
                row["rate"],
                errors="coerce"
            )

            if pd.isna(rating):

                rating_text = "N/A"

            else:

                rating_text = f"{rating:.1f}"


            # -------------------------------------------------
            # VOTES
            # -------------------------------------------------

            votes = pd.to_numeric(
                row["votes"],
                errors="coerce"
            )

            if pd.isna(votes):

                votes = 0

            votes = int(votes)


            # -------------------------------------------------
            # SIMILARITY
            # -------------------------------------------------

            similarity = float(
                row["Similarity"]
            )


            # =================================================
            # IMAGE + DETAILS
            # =================================================

            image_column, info_column = st.columns(
                [0.42, 0.58],
                gap="medium"
            )


            # -------------------------------------------------
            # FOOD IMAGE
            # -------------------------------------------------

            with image_column:

                st.image(
                    image_url,
                    use_container_width=True
                )


            # -------------------------------------------------
            # RESTAURANT DETAILS
            # -------------------------------------------------

            with info_column:

                st.markdown(
                    f'<div class="restaurant-name">'
                    f'🍴 {row["name"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )


                st.markdown(
                    f'<div class="restaurant-info">'
                    f'📍 {row["location"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )


                st.markdown(
                    f'<div class="restaurant-info">'
                    f'🍜 {row["cuisines"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )


                st.markdown(
                    f'<div class="restaurant-info">'
                    f'⭐ {rating_text} &nbsp; '
                    f'👥 {votes} votes'
                    f'</div>',
                    unsafe_allow_html=True
                )


                st.markdown(
                    f'<div class="match">'
                    f'🎯 {similarity:.0%} Match'
                    f'</div>',
                    unsafe_allow_html=True
                )


            # -------------------------------------------------
            # SEPARATOR
            # -------------------------------------------------

            st.divider()


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<p style="text-align:center;'
    'color:#78716c;'
    'font-size:12px;'
    'margin-top:45px;">'
    'Powered by TF-IDF & Cosine Similarity 🍽️'
    '</p>',
    unsafe_allow_html=True
)


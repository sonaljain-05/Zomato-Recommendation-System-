
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Zomato Recommendation",
    page_icon="🍽️",
    layout="wide"
)


# =====================================================
# CSS
# =====================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #000D00;
        color: white;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 30px;
        padding-bottom: 50px;
    }

    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: white;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #a1a1aa;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .left-box {
        background-color: #202023;
        border: 1px solid #3f3f46;
        border-radius: 20px;
        padding: 20px;
    }

    .left-heading {
        font-size: 28px;
        font-weight: 700;
        color: white;
        margin-top: 15px;
    }

    .left-text {
        color: #a1a1aa;
        font-size: 14px;
        line-height: 1.7;
        margin-bottom: 20px;
    }

    .recommend-title {
        font-size: 26px;
        font-weight: 700;
        color: white;
        margin-bottom: 5px;
    }

    .recommend-subtitle {
        color: #a1a1aa;
        font-size: 14px;
        margin-bottom: 20px;
    }

    .restaurant-name {
        font-size: 19px;
        font-weight: 700;
        color: white;
        margin-bottom: 5px;
    }

    .restaurant-info {
        color: #d4d4d8;
        font-size: 13px;
        margin: 5px 0;
    }

    .match {
        color: #fbbf24;
        font-weight: 700;
        font-size: 14px;
        margin-top: 8px;
    }

    div.stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 12px;
        border: none;
        background-color: #ef4444;
        color: white;
        font-size: 16px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #dc2626;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# HEADER
# =====================================================


st.markdown(
    '<div class="main-title" style="font-size: 40px;">🍽️ Zomato AI Recommendation</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="subtitle">'
    'Discover restaurants similar to the ones you already love.'
    '</div>',
    unsafe_allow_html=True
)


# =====================================================
# LOAD RESTAURANT DATA
# =====================================================

try:

    df = pd.read_pickle(
        "restaurant_data_small.pkl"
    )

except FileNotFoundError:

    st.error(
        "❌ restaurant_data_small.pkl not found।"
    )

    st.stop()


# =====================================================
# LOAD TF-IDF MATRIX
# =====================================================

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
        "❌ tfidf_matrix.npz not found।"
    )

    st.stop()


# =====================================================
# CLEAN DATA
# =====================================================

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


# =====================================================
# FOOD IMAGE LIST
#
# IMPORTANT:
# First 5 recommendations get 5 DIFFERENT images.
# 6th recommendation starts repeating from image 1.
# =====================================================

food_images = [

    # 1. Pizza
    "https://images.unsplash.com/"
    "photo-1574071318508-1cdbab80d002"
    "?auto=format&fit=crop&w=800&q=85",

    # 2. Burger
    "https://images.unsplash.com/"
    "photo-1568901346375-23c9450c58cd"
    "?auto=format&fit=crop&w=800&q=85",

    # 3. Biryani
    "https://images.unsplash.com/"
    "photo-1563379091339-03246963d51a"
    "?auto=format&fit=crop&w=800&q=85",

    # 4. South Indian
    "https://images.unsplash.com/"
    "photo-1630383249896-424e482df921"
    "?auto=format&fit=crop&w=800&q=85",

    # 5. Chinese
    "https://images.unsplash.com/"
    "photo-1563245372-f21724e3856d"
    "?auto=format&fit=crop&w=800&q=85",

    # 6. Sandwich
    "https://images.unsplash.com/"
    "photo-1528735602780-2552fd46c7af"
    "?auto=format&fit=crop&w=800&q=85",

    # 7. Dessert
    "https://images.unsplash.com/"
    "photo-1551024506-0bccd828d307"
    "?auto=format&fit=crop&w=800&q=85",

    # 8. Cafe
    "https://images.unsplash.com/"
    "photo-1501339847302-ac426a4a7cbb"
    "?auto=format&fit=crop&w=800&q=85",

    # 9. Restaurant
    "https://images.unsplash.com/"
    "photo-1517248135467-4c7edcad34c4"
    "?auto=format&fit=crop&w=800&q=85",

    # 10. Pasta
    "https://images.unsplash.com/"
    "photo-1473093295043-cdd812d0e601"
    "?auto=format&fit=crop&w=800&q=85"
]


# =====================================================
# TWO COLUMN LAYOUT
# =====================================================

left_column, right_column = st.columns(
    [0.9, 1.1],
    gap="large"
)


# =====================================================
# LEFT SIDE
# =====================================================

with left_column:

    st.markdown(
        '<div class="left-box">',
        unsafe_allow_html=True
    )

    st.image(
        "https://images.unsplash.com/"
        "photo-1517248135467-4c7edcad34c4"
        "?auto=format&fit=crop&w=900&q=90",
        use_container_width=True
    )

    st.markdown(
        '<div class="left-heading">'
        'Find Your Next Favourite 🍴'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="left-text">'
        'Select a restaurant you already love and let our '
        'recommendation system find similar restaurants for you.'
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


# =====================================================
# RIGHT SIDE
# =====================================================

with right_column:

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


    # =================================================
    # RESTAURANT SELECTBOX
    # =================================================

    restaurant_names = sorted(
        df["name"].unique()
    )

    selected_restaurant = st.selectbox(
        "Select Restaurant",
        restaurant_names
    )


    # =================================================
    # BUTTON
    # =================================================

    recommend_button = st.button(
        "✨ Show Recommendations"
    )


    # =================================================
    # RECOMMENDATIONS
    # =================================================

    if recommend_button:

        # Find selected restaurant
        selected_index = df.index[
            df["name"] == selected_restaurant
        ][0]


        # ---------------------------------------------
        # COSINE SIMILARITY
        # ---------------------------------------------

        similarity_scores = cosine_similarity(
            tfidf_matrix[selected_index],
            tfidf_matrix
        ).flatten()


        # ---------------------------------------------
        # CREATE RESULT
        # ---------------------------------------------

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


        # ---------------------------------------------
        # TOP 10 RECOMMENDATIONS
        # ---------------------------------------------

        result = result.head(10)


        st.markdown(
            "### ✨ Recommended For You"
        )


        # =================================================
        # DISPLAY RESTAURANTS
        # =================================================

        for recommendation_number, (_, row) in enumerate(
            result.iterrows()
        ):

            # ---------------------------------------------
            # IMAGE SELECTION
            #
            # % len() makes images repeat AFTER all
            # unique images have been used.
            #
            # 1st restaurant -> image 1
            # 2nd restaurant -> image 2
            # 3rd restaurant -> image 3
            # 4th restaurant -> image 4
            # 5th restaurant -> image 5
            # 6th restaurant -> image 6
            # ...
            # 11th restaurant -> image 1 again
            # ---------------------------------------------

            image_index = (
                recommendation_number
                % len(food_images)
            )

            image_url = food_images[
                image_index
            ]


            # ---------------------------------------------
            # RATING
            # ---------------------------------------------

            rating = pd.to_numeric(
                row["rate"],
                errors="coerce"
            )

            if pd.isna(rating):

                rating_text = "N/A"

            else:

                rating_text = f"{rating:.1f}"


            # ---------------------------------------------
            # VOTES
            # ---------------------------------------------

            votes = pd.to_numeric(
                row["votes"],
                errors="coerce"
            )

            if pd.isna(votes):

                votes = 0

            votes = int(votes)


            # ---------------------------------------------
            # SIMILARITY
            # ---------------------------------------------

            similarity = float(
                row["Similarity"]
            )


            # ---------------------------------------------
            # IMAGE + INFORMATION
            # ---------------------------------------------

            image_column, info_column = st.columns(
                [0.42, 0.58],
                gap="medium"
            )


            # ---------------------------------------------
            # IMAGE
            # ---------------------------------------------

            with image_column:

                st.image(
                    image_url,
                    use_container_width=True
                )


            # ---------------------------------------------
            # RESTAURANT INFORMATION
            # ---------------------------------------------

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


            # ---------------------------------------------
            # SEPARATOR
            # ---------------------------------------------

            st.divider()


# =====================================================
# FOOTER
# =====================================================

st.markdown(
    '<p style="text-align:center;'
    'color:#666;'
    'font-size:12px;">'
    'Powered by TF-IDF & Cosine Similarity 🍽️'
    '</p>',
    unsafe_allow_html=True
)



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
        background-color: #111111;
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
    '<div class="main-title">🍽️ Zomato AI Recommendation</div>',
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
        "❌ restaurant_data_small.pkl नहीं मिली।"
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
        "❌ tfidf_matrix.npz नहीं मिली।"
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
    "?auto=format&fit=crop&
```

```text
import streamlit as st
import pandas as pd
import numpy as np
import os

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Zomato Restaurant Finder",
    page_icon="🍴",
    layout="centered"
)


st.markdown(
    """
    <style>

    .stApp {
        background: #18181b;
        color: #f5f5f5;
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
        color: #a1a1aa;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .hero-image {
        display: block;
        width: 100%;
        max-width: 700px;
        height: 260px;
        object-fit: cover;
        margin: 0 auto 25px auto;
        border-radius: 18px;
    }

    .restaurant-card {
        background: #27272a;
        border: 1px solid #3f3f46;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 18px;
    }

    .restaurant-name {
        color: white;
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .restaurant-info {
        color: #d4d4d8;
        font-size: 14px;
        line-height: 1.7;
    }

    .score {
        color: #fbbf24;
        font-weight: 700;
    }

    .stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 12px;
        font-size: 17px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


if not os.path.exists("restaurant_data_small.pkl"):
    st.error("restaurant_data_small.pkl file not found!")
    st.stop()


if not os.path.exists("tfidf_matrix.npz"):
    st.error("tfidf_matrix.npz file not found!")
    st.stop()


df = pd.read_pickle("restaurant_data_small.pkl")


try:

    z = np.load(
        "tfidf_matrix.npz",
```

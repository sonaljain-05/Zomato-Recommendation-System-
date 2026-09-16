```python
import streamlit as st
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Zomato Recommendation System",
    page_icon="🍴",
    layout="centered"
)


# --------------------------------------------------
# CUSTOM DARK THEME
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #18181b;
        color: #f5f5f5;
    }

    /* Keep app compact */
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #a1a1aa;
        font-size: 16px;
        margin-bottom: 25px;
    }

    /* Hero image */
    .hero-image {
        display: block;
        width: 100%;
        max-width: 700px;
        height: 260px;
        object-fit: cover;
        margin: 0 auto 25px auto;
        border-radius: 18px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
    }

    /* Selectbox label */
    label {
        color: #e4e4e7 !important;
        font-weight: 600 !important;
    }

    /* Recommendation cards */
    .restaurant-card {
        background: #27272a;
        border: 1px solid #3f3f46;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 18px;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.25);
    }

    .restaurant-name {
        color: #ffffff;
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

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 50px;
        font-size: 17px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe
```

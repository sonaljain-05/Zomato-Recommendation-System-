
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


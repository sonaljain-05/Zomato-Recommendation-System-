```python
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Zomato Recommendation",
    page_icon="🍽️",
    layout="wide"
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
    background: #111113;
    color: white;
}

.block-container {
    max-width: 1200px;
    padding-top: 35px;
    padding-bottom: 50px;
}


/* ================= LEFT SIDE ================= */

.left-panel {
    background: linear-gradient(
        145deg,
        #29292d,
        #18181b
    );

    border-radius: 25px;
    padding: 25px;
    min-height: 650px;
    border: 1px s
```

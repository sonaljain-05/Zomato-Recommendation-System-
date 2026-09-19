import streamlit as st
import pandas as pd
import numpy as np
import os

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================

# PAGE CONFIGURATION

# ==================================================

st.set_page_config(
page_title="Zomato Recommendation System",
page_icon="🍽️",
layout="centered"
)

# ==================================================

# CUSTOM CSS

# ==================================================

st.markdown(
""" <style>

```
.stApp {
    background-color: #18181b;
    color: white;
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
    color: #bdbdbd;
    font-size: 16px;
    margin-bottom: 25px;
}

.hero-box {
    background-color: #27272a;
    border-radius: 18px;
    padding: 10px;
    margin-bottom: 25px;
}

.hero-box img {
    width: 100%;
    height: 250px;
    object-fit: cover;
    border-radius: 14px;
}

.restaurant-card {
    background-color: #27272a;
    border: 1px solid #3f3f46;
    border-radius: 15px;
    padding: 18px;
    margin-top: 15px;
    margin-bottom: 15px;
}

.restaurant-name {
    font-size: 21px;
    font-weight: 700;
    color: white;
    margin-bottom: 8px;
}

.restaurant-info {
    color: #d4d4d8;
    font-size: 14px;
    line-height: 1.8;
}

.score {
    color: #fbbf24;
    font-weight: 600;
}

</style>
""",
unsafe_allow_html=True
```

)

# ==================================================

# TITLE

# ==================================================

st.markdown(
'<div class="main-title">🍽️ Zomato Recommendation System</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">Find restaurants similar to your choice</div>',
unsafe_allow_html=True
)

# ==================================================

# HERO IMAGE

# ==================================================

st.markdown(
""" <div class="hero-box"> <img
         src="https://images.unsplash.com/photo-1517248135467-4c7edcad34c4"
         alt="Restaurant"
     > </div>
""",
unsafe_allow_html=True
)

# ==================================================

# LOAD RESTAURANT DATA

# ==================================================

@st.cache_data
def load_restaurant_data():

```
file_path = "restaurant_data_small.pkl"

if not os.path.exists(file_path):
    return None

data = pd.read_pickle(file_path)

return data
```

# ==================================================

# LOAD TF-IDF MATRIX

# ==================================================

@st.cache_resource
def load_tfidf_matrix():

```
file_path = "tfidf_matrix.npz"

if not os.path.exists(file_path):
    return None

z = np.load(
    file_path,
    allow_pickle=False
)

data = z["data"]
indices = z["indices"]
indptr = z["indptr"]
shape = tuple(z["shape"])

matrix = csr_matrix(
    (data, indices, indptr),
    shape=shape
)

return matrix
```

# =================

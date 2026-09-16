import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import os


# -----------------------------
# Page Settings
# -----------------------------

st.set_page_config(
    page_title="Zomato Recommendation System",
    page_icon="🍴",
    layout="wide"
)


# -----------------------------
# Check Files
# -----------------------------

if not os.path.exists("restaurant_data_small.pkl"):
    st.error("restaurant_data_small.pkl file not found!")
    st.stop()

if not os.path.exists("tfidf_matrix.npz"):
    st.error("tfidf_matrix.npz file not found!")
    st.stop()


# -----------------------------
# Load Restaurant Data
# -----------------------------

df = pd.read_pickle("restaurant_data_small.pkl")


# -----------------------------
# Load TF-IDF Matrix Manually
# -----------------------------

try:

    npz = np.load(
        "tfidf_matrix.npz",
        allow_pickle=False
    )

    data = npz["data"]
    indices = npz["indices"]
    indptr = npz["indptr"]
    shape = tuple(npz["shape"])

    tfidf_matrix = csr_matrix(
        (data, indices, indptr),
        shape=shape
    )

except Exception as e:

    st.error("TF-IDF matrix could not be loaded.")
    st.code(str(e))
    st.stop()


# -----------------------------
# Check Data and Matrix
# -----------------------------

if tfidf_matrix.shape[0] != len(df):

    st.error(
        f"Data mismatch: TF-IDF matrix has "
        f"{tfidf_matrix.shape[0]} rows but restaurant data has "
        f"{len(df)} rows."
    )

    st.stop()


# -----------------------------
# Title
# -----------------------------

st.title("🍴 Zomato Restaurant Recommendation System")

st.write(
    "Find similar restaurants based on cuisines, "
    "location, restaurant type and customer reviews."
)


# -----------------------------
# Restaurant Selection
# -----------------------------

restaurant_names = sorted(
    df["name"].dropna().unique()
)

selected_restaurant = st.selectbox(
    "Select a Restaurant",
    restaurant_names
)


# -----------------------------
# Recommendation Function
# -----------------------------

def recommend(name):

    matches = df[
        df["name"].str.lower() == name.lower()
    ]

    if len(matches) == 0:
        return pd.DataFrame()

    index = matches.index[0]

    scores = cosine_similarity(
        tfidf_matrix[index],
        tfidf_matrix
    )[0]

    top_indices = scores.argsort()[-11:][::-1]

    top_indices = [
        i for i in top_indices
        if i != index
    ][:10]

    result = df.iloc[top_indices].copy()

    result["Similarity Score"] = scores[top_indices]

    return result[
        [
            "name",
            "location",
            "cuisines",
            "rate",
            "votes",
            "Similarity Score"
        ]
    ]


# -----------------------------
# Recommendation Button
# -----------------------------

if st.button("🔍 Get Recommendations"):

    recommendations = recommend(
        selected_restaurant
    )

    st.subheader(
        f"Recommended Restaurants for {selected_restaurant}"
    )

    if len(recommendations) > 0:

        for _, row in recommendations.iterrows():

            st.markdown("---")

            st.subheader(row["name"])

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    "📍 **Location:**",
                    row["location"]
                )

                st.write(
                    "🍽️ **Cuisines:**",
                    row["cuisines"]
                )

            with col2:

                st.write(
                    "⭐ **Rating:**",
                    row["rate"]
                )

                st.write(
                    "👍 **Votes:**",
                    row["votes"]
                )

                st.write(
                    "📊 **Similarity:**",
                    round(
                        row["Similarity Score"],
                        3
                    )
                )

    else:

        st.warning(
            "Restaurant not found."
        )

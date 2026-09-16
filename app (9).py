import streamlit as st
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Load Data
# -----------------------------

df = pd.read_pickle("restaurant_data_small.pkl")

tfidf_matrix = load_npz("tfidf_matrix.npz")


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Zomato Recommendation System",
    page_icon="🍴",
    layout="wide"
)


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
        "Recommended Restaurants"
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
                    "📊 **Similarity Score:**",
                    round(row["Similarity Score"], 3)
                )

    else:

        st.warning(
            "Restaurant not found."
        )
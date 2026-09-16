
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Zomato Recommendation",
    page_icon="🍴",
    layout="centered"
)

# ---------- DATA ----------
df = pd.read_pickle("restaurant_data_small.pkl")

z = np.load("tfidf_matrix.npz", allow_pickle=False)

tfidf = csr_matrix(
    (z["data"], z["indices"], z["indptr"]),
    shape=tuple(z["shape"])
)

# ---------- HEADER ----------
st.title("🍴 Zomato Restaurant Finder")
st.caption("Find restaurants similar to your favourite place")

# ---------- RESTAURANT SELECT ----------
names = sorted(df["name"].dropna().unique())

selected = st.selectbox(
    "Select Restaurant",
    names
)

# ---------- BUTTON ----------
if st.button(
    "🔍 Find Similar Restaurants",
    type="primary",
    width="stretch"
):

    index = df[
        df["name"].str.lower() == selected.lower()
    ].index[0]

    scores = cosine_similarity(
        tfidf[index],
        tfidf
    )[0]

    top = scores.argsort()[-11:][::-1]
    top = [i for i in top if i != index][:10]

    st.subheader("Recommended Restaurants")

    images = [
        "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38",
        "https://images.unsplash.com/photo-1563379926898-05f4575a45d8",
        "https://images.unsplash.com/photo-1555939594-58d7cb561ad1",
        "https://images.unsplash.com/photo-1513104890138-7c749659a591",
        "https://images.unsplash.com/photo-1547592180-85f173990554"
    ]

    for row in range(0, len(top), 2):

        col1, col2 = st.columns(2)

        for col, i in zip(
            [col1, col2],
            top[row:row + 2]
        ):

            r = df.iloc[i]

            with col:

                st.image(
                    images[i % len(images)],
                    width=300
                )

                st.markdown(
                    f"### 🍴 {r['name']}"
                )

                st.write(
                    f"📍 {r['location']}"
                )

                st.write(
                    f"🍛 {r['cuisines']}"
                )

                st.write(
                    f"⭐ {r['rate']}   •   👍 {r['votes']:,}"
                )

                st.caption(
                    f"AI Similarity: {scores[i]:.1%}"
                )

                st.divider()


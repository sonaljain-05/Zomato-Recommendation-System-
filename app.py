import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set Streamlit page configuration
st.set_page_config(
    page_title="Zomato Restaurant Recommender",
    page_icon="🍽️",
    layout="wide"
)

# Title and Subheading
st.title("🍽️ Zomato Restaurant Recommendation System")
st.markdown("Find restaurants similar to your favorites based on cuisines, location, and ratings!")

# Load Data and Similarity Matrix
@st.cache_data
def load_data():
    df = pd.read_csv('zomato_cleaned.csv')
    return df

@st.cache_resource
def load_similarity_matrix():
    with open('cosine_sim.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)
    return cosine_sim

try:
    df = load_data()
    cosine_sim = load_similarity_matrix()
    
    # Create mapping of restaurant names to indices
    indices = pd.Series(df.index, index=df['name']).drop_duplicates()

    # Recommendation Logic
    def get_recommendations(title, top_n=5):
        if title not in indices:
            return None
        
        idx = indices[title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
            
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        
        restaurant_indices = [i[0] for i in sim_scores]
        
        recommendations = df.iloc[restaurant_indices][
            ['name', 'location', 'cuisines', 'rate', 'cost']
        ]
        return recommendations

    # Sidebar Filter Controls
    st.sidebar.header("Filter & Search Options")
    selected_restaurant = st.sidebar.selectbox(
        "Select or search a restaurant:",
        df['name'].unique()
    )

    top_k = st.sidebar.slider("Number of recommendations:", min_value=1, max_value=10, value=5)

    # Main App Logic
    if st.sidebar.button("Get Recommendations"):
        st.subheader(f"Recommendations similar to: **{selected_restaurant}**")
        
        recommendations = get_recommendations(selected_restaurant, top_n=top_k)
        
        if recommendations is not None and not recommendations.empty:
            cols = st.columns(min(len(recommendations), 3))
            
            for i, (_, row) in enumerate(recommendations.iterrows()):
                col_idx = i % 3
                with cols[col_idx]:
                    st.write(f"### {row['name']}")
                    st.write(f"📍 **Location:** {row['location']}")
                    st.write(f"🍕 **Cuisines:** {row['cuisines']}")
                    st.write(f"⭐ **Rating:** {row['rate']}")
                    st.write(f"💰 **Cost for Two:** ₹{row['cost']}")
                    st.divider()
        else:
            st.error("No recommendations found for the selected restaurant.")

    # Additional Exploratory View
    with st.expander("🔍 View Raw Restaurant Dataset"):
        st.dataframe(df.head(20))

except FileNotFoundError:
    st.error("Data or model files not found. Please place `zomato_cleaned.csv` and `cosine_sim.pkl` in the root folder.")

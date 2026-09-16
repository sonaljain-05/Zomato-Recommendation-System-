import streamlit as st
import os

st.title("TF-IDF File Check")

file_name = "tfidf_matrix.npz"

if not os.path.exists(file_name):
    st.error("tfidf_matrix.npz NOT FOUND")
    st.stop()

# File size
file_size = os.path.getsize(file_name)

st.write("File size:", file_size, "bytes")
st.write("File size:", round(file_size / (1024 * 1024), 2), "MB")


# Read first few bytes
with open(file_name, "rb") as f:
    first_bytes = f.read(100)

st.write("First bytes of file:")
st.code(str(first_bytes))

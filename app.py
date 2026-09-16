import streamlit as st
import numpy as np
import zipfile

st.title("TF-IDF NPZ Check")

file_name = "tfidf_matrix.npz"

try:
    with zipfile.ZipFile(file_name, "r") as z:

        st.write("Files inside NPZ:")
        st.write(z.namelist())

        for file in z.namelist():
            st.write(f"Checking: {file}")

            with z.open(file) as f:
                data = np.load(f, allow_pickle=True)

                st.write("dtype:", data.dtype)
                st.write("shape:", data.shape)

except Exception as e:

    st.error("NPZ inspection failed")
    st.code(str(e))

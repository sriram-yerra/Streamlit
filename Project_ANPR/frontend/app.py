import streamlit as st
import requests

st.title("AI Vision Dashboard")

API_ENDPOINT = "https://8000-humble-funicular-7v4969w9972p644.app.github.dev"

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file and st.button("Run Detection"):
    try:
        response = requests.post(
            f"{API_ENDPOINT}/detect-image",
            files={"file": uploaded_file.getvalue()},
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()

            download_link = (
                data["download_url"]
                if data["download_url"].startswith("http")
                else API_ENDPOINT + data["download_url"]
            )

            st.success("Detection complete")
            st.markdown(f"[⬇ Download Result Image]({download_link})")

        else:
            st.error("API Error")

    except requests.exceptions.RequestException as e:
        st.error(f"Connection failed: {e}")

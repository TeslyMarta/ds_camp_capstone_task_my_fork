# Thirdparty imports
import requests
import streamlit as st

# Local imports
from config import APIConfig

# Set the page configuration for Streamlit
st.set_page_config(page_title="Summarizer", layout="centered", initial_sidebar_state="collapsed")

# Display the title of the application
st.title("📜 Text Summarizer")

# Create a text area for the user to enter the text to be summarized
text = st.text_area("Enter text to summarize:", height=200)

# Check if the "Summarize Text" button is clicked
if st.button("Summarize Text"):

    # Check if the text area is not empty
    if text.strip():

        # Send a POST request to the FastAPI server with the text to be summarized
        api_url = f"{APIConfig.HOST}/summarize/"
        payload = {"text": text}
        response = requests.post(api_url, json=payload)

        # Check if the request is successful (status code 200)
        if response.status_code == 200:

            # Get the summarized text from the response
            summary = response.json().get("summary", "No summary provided.")

            # Display the summarized text as a subheader
            st.subheader("Result:")
            st.write(summary)
        else:
            # Display an error message if the request fails
            st.error("Failed to summarize text. Please check the API.")
    else:
        # Display a warning message if the text area is empty
        st.warning("Please enter text to summarize.")

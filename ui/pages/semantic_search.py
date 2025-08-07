# Thirdparty imports
import requests
import streamlit as st

# Local imports
from config import APIConfig

st.title("Semantic Search for Products")
st.markdown("Describe the product you are looking for, and we will find the best matches for you!")

query = st.text_input("Product description", placeholder="for example, 'red running shoes with good grip'")

if st.button("Find Products", use_container_width=True, type="primary"):
    if query.strip():
        with st.spinner("Searching for products..."):
            try:
                api_url = f"{APIConfig.HOST}/semantic_search/"
                payload = {"text": query}

                response = requests.post(api_url, json=payload)

                if response.status_code == 200:
                    search_data = response.json()
                    results = search_data.get("results", [])

                    if not results:
                        st.warning("Unfortunately, no matching products were found.")
                    else:
                        st.subheader("Here are the top matches for your query:")
                        for product in results:
                            st.markdown(f"**{product['rank']}. {product['title']}**")
                            st.write(f"**Brand:** {product['brand']} | **Category:** {product['category']}")
                            st.write(f"_{product['description']}_")
                            st.info(f"**Similarity:** {product['similarity']:.4f}")
                            st.divider()

                elif response.status_code == 404:
                    st.warning("No matches found for your query.")
                else:
                    st.error(f"Error: {response.status_code}")
                    st.json(response.json())
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred while connecting to the API: {e}")
    else:
        st.warning("Please enter a product description to search for.")

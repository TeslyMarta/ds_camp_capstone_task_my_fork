# Thirdparty imports
import streamlit as st

st.set_page_config(
    page_title="General page",
    page_icon="👋",
)

st.title("Welcome")
st.sidebar.success("Choose a page above.")

st.markdown(
    """
    Here you can find two examples of using the models from the course:
    """
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Semantic search", use_container_width=True, type="primary"):
        st.switch_page("pages/semantic_search.py")

with col2:
    if st.button("Text summarization", use_container_width=True, type="primary"):
        st.switch_page("pages/base_chat.py")

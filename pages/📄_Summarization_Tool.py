import streamlit as st
from base import request_gpt

st.set_page_config(
    page_title="AI Summary", 
    page_icon="🤖",
)

st.write("# Summarization Tool 📄")
    
# upload file by streamlit
uploaded_file = st.file_uploader("Upload file")

if not uploaded_file:
    st.stop()    

option = st.selectbox(
   "What type of document do you want to summarize?",
   ("Default", "Book", "Paper", "Article"),
    index=None
)

if not option:
    st.stop()

if option == "Default":
    prompt = """Provide a summary of the document."""

elif option == "Book":
    chapter = st.text_input("Which chapter do you want to summarize?")
    if not chapter:
        prompt = """Provide a summary of the book."""
    else:
        prompt = f"""Provide a summary of chapter {chapter} the book."""

elif option == "Paper":
    option_2 = st.selectbox(
        "What part do you want to focus on?",
        ("Default", "Abstract", "Introduction", "Methodology", "Results", "Conclusion"),
        index=None)
    
    if not option_2:
        st.stop()
    
    if option_2 == "Default":
        prompt = f"""
        Provide a long, detailed summary of the paper with the following format:
            1. Authors:
            2. Published date:
            3. Title:
            4. Abstract:
            5. Introduction:
            6. Methodology:
            7. Results:
            8. Conclusion:
            9. References:
        Must be very precise, must not make up words, use the words in the paper.
        """
    else:
        prompt = f"""
        Provide a long, detailed summary of the paper with the following format:
            1. Authors:
            2. Published date:
            3. Title:
            4. Abstract:
            5. Introduction:
            6. Methodology:
            7. Results:
            8. Conclusion:
            9. References:
        Must be very detailed in part {option_2}, be brief in other parts.
        Must be very precise, must not make up words, use the words in the paper.
        """

elif option == "Article":
    prompt = """Provide a summary of the article."""

limit_words = st.selectbox("Limit the summary by words?", ('Yes', 'No'), index=None)

if not limit_words:
    st.stop()

if limit_words == 'Yes':
    num_words = st.text_input("How many words do you want the summary to be?")
    if not num_words:
        st.stop()
    prompt += f" The summary should be about {num_words} words long."

# st.markdown(prompt)

if st.button("Summarize"):
    request_gpt(uploaded_file, prompt, "summary")
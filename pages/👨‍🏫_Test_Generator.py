import streamlit as st
from io import BytesIO
from base import request_gpt
import PyPDF2

st.set_page_config(
    page_title="Test Generator", 
    page_icon="🤖",
)

st.write("# Test Generator 👨‍🏫")

uploaded_file = st.file_uploader("Upload file")

if not uploaded_file:
    st.stop()

num_question = st.selectbox(
   "Choose number of question:",
    ("5", "10", "15", "20"),
     index=None
)

if not num_question:
    st.stop()

question_type = st.selectbox(
    "Choose question type:",
     ("Multiple choice", "True/False", "Fill in the blank"),
      index=None
    )

if not question_type:
    st.stop()

difficulty = st.selectbox(
    "Choose difficulty level:",
     ("Easy", "Medium", "Hard"),
      index=None
    )

if not difficulty:
    st.stop()

limit = st.selectbox(
    "Limit the content in the file?",
    ('Yes', 'No'),
    index=None
)

if not limit:
    st.stop()

prompt = f"From the input file, generate a test with {num_question} questions of type {question_type} with difficulty level {difficulty}."

if limit == 'Yes':
    limit_type = st.selectbox(
        "Choose how to limit content:",
        ('By Page Range', 'By Chapters'),
        index=None
    )

    if not limit_type:
        st.stop()

    if limit_type == 'By Page Range':
        pdfReader = PyPDF2.PdfFileReader(BytesIO(uploaded_file.read()))
        num_pages = pdfReader.getNumPages()

        limit_range = st.slider("Choose the page range of the file:", 1, num_pages, (1, num_pages))
        st.markdown(f"Content range: {limit_range[0]} - {limit_range[1]}")
        extra_prompt = f" Limit the content in the file from page {limit_range[0]} to page {limit_range[1]}."

    elif limit_type == 'By Chapters':
        chapters = st.text_area("Enter the chapter numbers or titles")
        if not chapters:
            st.warning("Please enter at least one chapter number or title.")
            st.stop()
        extra_prompt = f" Limit the content to chapter {chapters} in the file."


    prompt = prompt + extra_prompt

prompt = prompt + " At the end of the test, add a new section to provide the correct answer and detailed explanation for each question. Also refer to the page number in the file for each question and quote the relevant text."

# st.markdown(prompt)

if st.button("Generate Test"):
    request_gpt(uploaded_file, prompt, "test")
    
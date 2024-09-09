import streamlit as st

st.set_page_config(
    page_title="Quant Retail AI",
    page_icon="🤖",
)

st.write("# Quant Retail AI 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    #### Giới thiệu: 
    -    Bản Beta Quant Retail AI được xây dựng với mục đích thử nghiệm.
    -    Các chức năng hiện có:
            + Tóm tắt văn bản.
            + Kiểm tra và giải thích code. 
            + Viết kết luận Báo cáo giám sát tự động.
            + Chatbot về Basel.
            + Tạo bài kiểm tra từ tài liệu.
            \n
    #### Hướng dẫn sử dụng:\n
    - B1: Chọn chức năng muốn sử dụng
    - B2: Upfile theo chức năng từ máy tính/điện thoại
    - B3: Download file kết quả\n
    Powered by: OpenAI, Streamlit
"""
)

# st.markdown(
#     """
#     Streamlit is an open-source app framework built specifically for
#     Machine Learning and Data Science projects.
#     **👈 Select a demo from the sidebar** to see some examples
#     of what Streamlit can do!
#     ### Want to learn more?
#     - Check out [streamlit.io](https://streamlit.io)
#     - Jump into our [documentation](https://docs.streamlit.io)
#     - Ask a question in our [community
#         forums](https://discuss.streamlit.io)
#     ### See more complex demos
#     - Use a neural net to [analyze the Udacity Self-driving Car Image
#         Dataset](https://github.com/streamlit/demo-self-driving)
#     - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
# """
# )
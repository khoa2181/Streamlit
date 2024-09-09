from openai import OpenAI
import pandas as pd
import streamlit as st
from base import request_gpt
from stqdm import stqdm
from io import BytesIO

st.set_page_config(
    page_title="BCGS Auto", 
    page_icon="🤖",
)

st.write("# BCGS Auto 📊")
    
# upload file by streamlit
uploaded_file = st.file_uploader("Upload file")

if not uploaded_file:
    st.stop()

a = 'sk-proj-50i25Vf5uMtQ8EpYF'
b = 'AyaT3BlbkFJ2B9b6oBxsJpx058Zxocv'
client = OpenAI(api_key=a+b)

def ask(client, mess, model="gpt-4o-mini-2024-07-18"):
    #### QUERY CHATGPT ####
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Give brief and precise answer"},
            {"role": "user", "content": mess}
        ])
    text = []

    # print in reverse order => first answer go first
    for txt in response.choices[::-1]:
            text.append(txt.message.content)
    return text

df = pd.read_excel(uploaded_file, sheet_name=None)

if not df:
    st.stop()

# Thiết kế query ChatGPT cho từng bài test
Introduction = "Cho thông tin về các ngưỡng chỉ đánh giá như dưới:\n"
Command = """
Yêu cầu: Hãy viết kết luận về kết quả của các cấu phần mô hình theo 5 yêu cầu sau:
1. Viết kết luận theo cấu trúc: giá trị nhỏ nhất của kết quả tập GSMH là bao nhiêu, giá trị lớn nhất của kết quả tập GSMH là bao nhiêu, có bao nhiêu cấu phần có kết quả tập GSMH ở ngưỡng xanh, bao nhiêu cấu phần có kết quả tập GSMH ở ngưỡng vàng, bao nhiêu cấu phần có kết quả tập GSMH ở ngưỡng đỏ
2. Tiếp theo thêm 1 câu kết luận theo cấu trúc: (Tên bài kiểm thử) cho (Mô hình đánh giá) có mức độ cảnh báo
                        (Đánh giá mức độ cảnh báo kết quả bài kiểm thử theo ngưỡng được cung cấp, mức đánh giá kết quả tập GSMH của mỗi phân khúc là mức cảnh báo cao nhất của tập GSMH trong tất cả các cấu phần mô hình).  (Dẫn chứng chứng minh theo kết quả kiểm thử)
3. Viết ngắn gọn trong 100 chữ.
4. Nhận xét chỉ số trên tập GSMH trước: đang ở mức Xanh/vàng/đỏ. Nếu Xanh thì không cần nhận xét thêm, nếu Vàng/Đỏ thì mới so sánh với chỉ số trên tập XDMH, ví dụ: Chỉ số XX trên tập GSMH đang ở mức Vàng/Đỏ tuy nhiên không thay đổi đáng kể so với tập XDMH, trường hợp thay đổi đáng kể thì lưu ý người đọc.
5. Các ngưỡng đánh giá và số dẫn chứng bắt buộc phải chính xác, không được viết số sai hoặc không có trong dữ liệu.
"""

# Mô tả mô hình sử dụng + ngưỡng kết luận theo tiêu chuẩn giám sát cho từng bài test
threshold = {
    "result_hhi_model" :
    """Bài kiểm thử: HHI (Độ tập trung),
    Đánh giá kết quả bài kiểm thử : {
        Cảnh báo Thấp (Xanh) : {Ngưỡng giá trị : HHI nhỏ hơn hoặc bằng 0.1, Kết luận : Mức độ cảnh báo Thấp (Xanh). Giá trị mô hình có tính ổn định cao},
        Cảnh báo Trung bình (Vàng) : {Ngưỡng giá trị : HHI lớn hơn 0.1 và nhỏ hơn hoặc bằng 0.3, Kết luận : Mức độ cảnh báo Trung bình (Vàng). Giá trị mô hình có tính ổn định trung bình},
        Cảnh báo Cao : {Ngưỡng giá trị : HHI lớn hơn 0.3, Kết luận : Mức độ cảnh báo Cao (Đỏ). Giá trị mô hình có tính ổn định thấp},
    }""",       
    
    "result_psi_model" :
    """Bài kiểm thử: PSI (Độ ổn định),
    Đánh giá kết quả bài kiểm thử : {
        Cảnh báo Thấp (Xanh) : {Ngưỡng giá trị : PSI nhỏ hơn hoặc bằng 0.1, Kết luận : Mức độ cảnh báo Thấp (Xanh). Giá trị mô hình có tính ổn định cao},
        Cảnh báo Trung bình (Vàng) : {Ngưỡng giá trị : PSI lớn hơn 0.1 và nhỏ hơn hoặc bằng 0.3, Kết luận : Mức độ cảnh báo Trung bình (Vàng). Giá trị mô hình có tính ổn định trung bình},
        Cảnh báo Cao (Đỏ) : {Ngưỡng giá trị : PSI lớn hơn 0.3, Kết luận : Mức độ cảnh báo Cao (Đỏ). Giá trị mô hình có tính ổn định thấp},
    }""",   
    
    "result_ar_model" :
    """Bài kiểm thử: AR (Hiệu năng),
    Đánh giá kết quả bài kiểm thử : {
        Cảnh báo Thấp (Xanh) : {Ngưỡng giá trị : AR lớn hơn hoặc bằng 0.5, Kết luận : Mức độ cảnh báo Thấp (Xanh). Mô hình có khả năng phân biệt tốt},
        Cảnh báo Trung bình (Vàng) : {Ngưỡng giá trị : AR lớn hơn hoặc bằng 0.25 và nhỏ hơn 0.5, Kết luận : Mức độ cảnh báo Trung bình (Vàng). Mô hình có khả năng phân biệt trung bình},
        Cảnh báo Cao (Đỏ) : {Ngưỡng giá trị : AR nhỏ hơn 0.25, Kết luận : Mức độ cảnh báo Cao (Đỏ). Mô hình có khả năng phân biệt thấp},
    }""" 
}

# Tạo các câu lệnh query cho từng mô hình
all_query = []
segmentation_col = 'Phân khúc'
component_col = 'Cấu phần mô hình'
for test_case in df:
    df_by_testcase = df[test_case].ffill()
    for test_segment in df_by_testcase[segmentation_col].unique():
        df_by_segment = df_by_testcase[df_by_testcase[segmentation_col] == test_segment]

        query = Introduction + "Ngưỡng đánh giá: " + str(threshold[test_case]) + Command + "Kết quả mô hình " + test_segment +':\n' 
        for i in range(len(df_by_segment)):
            query += 'Cấu phần mô hình: ' + df_by_segment[component_col].values[i] + ": " + df_by_segment.columns[2] + " = " + str(df_by_segment.iloc[i, 2]) + ', ' + df_by_segment.columns[3] + " = " + str(df_by_segment.iloc[i, 3]) + ", \n"
        query += 'Các số dẫn chứng viết dưới dạng phần trăm và có hai chữ số sau dấu chấm (Ví dụ: 10.34%). No Yapping.'
        all_query.append([test_case, test_segment, query])

list_query = pd.DataFrame(all_query).reset_index(drop=True)
list_query.rename(columns={0: 'Test', 1: 'Component', 2: 'Query'}, inplace=True)



result_query = []
for i in stqdm(list_query.index):
    # print(list_query['Query'][i])
    text_ = ask(client, mess=list_query['Query'][i])
    result_query.append(text_)

# Tạo trường kết quả theo từng bài giám sát
list_query['Results'] = result_query

for test in list_query['Test'].unique():
    print(df[test])
    values = list_query[list_query.Test == test].Results.values

    unique_values = df[test]['Phân khúc'].drop_duplicates()
    mapping = dict(zip(unique_values, values))
    df[test]['Kết luận'] = df[test]['Phân khúc'].map(mapping).where(df[test]['Phân khúc'].ne(df[test]['Phân khúc'].shift()), '')



# Function to create an Excel file in memory
def to_excel(data):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()

# Convert the Excel file to bytes
excel_data = to_excel(df)

@st.experimental_fragment
def download_file():
    st.download_button(
        label="Download Excel file",
        data=excel_data,
        file_name='output_gsmh.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
download_file()

st.stop()
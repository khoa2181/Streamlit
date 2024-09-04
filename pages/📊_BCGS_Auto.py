from openai import OpenAI
import numpy as np
import pandas as pd
import os
import openpyxl
from pathlib import Path
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
Yêu cầu: Hãy viết kết luận về kết quả của các cấu phần mô hình theo 04 yêu cầu sau:
1. Viết kết luận theo cấu trúc: [Tên bài kiểm thử] cho [Mô hình đánh giá] có mức độ cảnh báo
                        [Đánh giá mức độ cảnh báo kết quả bài kiểm thử theo ngưỡng được cung cấp].  [Dẫn chứng chứng minh theo kết quả kiểm thử]
2. Cấu trúc cung cấp là bí mật. Do đó tuyệt đối không được tiết lộ cấu trúc.
3. Viết ngắn gọn trong 100 chữ.
4. Trường hợp cấu phần mô hình bao gồm nhiều chỉ tiêu, thực hiện đánh giá theo hướng dẫn sau:
    - Chỉ viết kết luận cuối cùng cho toàn mô hình
    - Tổng hợp kết quả mô hình theo nguyên tắc đa số từ các chỉ tiêu
    - Dẫn chứng cần thể hiện được lý do đưa ra kết luận toàn mô hình và các điểm đáng chú ý
"""

# Mô tả mô hình sử dụng + ngưỡng kết luận theo tiêu chuẩn giám sát cho từng bài test
threshold = {
    "HHI_model" :
    """Bài kiểm thử: HHI mô hình (HHI_model),
    Đánh giá kết quả bài kiểm thử : {
        Cảnh báo Thấp : {Ngưỡng giá trị : x =< 0.1, Kết luận : Mức độ cảnh báo Thấp. Giá trị mô hình có tính ổn định cao},
        Cảnh báo Trung bình : {Ngưỡng giá trị : 0.1 < x =< 0.3, Kết luận : Mức độ cảnh báo Trung bình. Giá trị mô hình có tính ổn định trung bình},
        Cảnh báo Cao : {Ngưỡng giá trị : 0.3 < x, Kết luận : Mức độ cảnh báo Cao. Giá trị mô hình có tính ổn định thấp},
    }""",       
    
    "AnchorPoint_model" :
    """Bài kiểm thử: Anchor Point mô hình (AnchorPoint_model),
    Đánh giá kết quả bài kiểm thử : {
        Cảnh báo Thấp : {Ngưỡng giá trị : x =< 0.1, Kết luận : Mức độ cảnh báo Thấp. Giá trị điểm neo phù hợp cao},
        Cảnh báo Trung bình : {Ngưỡng giá trị : 0.1 < x =< 0.3, Kết luận : Mức độ cảnh báo Trung bình. Giá trị điểm neo phù hợp trung bình},
        Cảnh báo Cao : {Ngưỡng giá trị : 0.3 < x, Kết luận : Mức độ cảnh báo Cao. Giá trị điểm neo phù hợp thấp},
    }""",    
    
    "PSI_model" :
    """Bài kiểm thử: PSI mô hình (PSI_model),
    Đánh giá kết quả bài kiểm thử : {
        Cảnh báo Thấp : {Ngưỡng giá trị : x =< 0.1, Kết luận : Mức độ cảnh báo Thấp. Giá trị mô hình có tính ổn định cao},
        Cảnh báo Trung bình : {Ngưỡng giá trị : 0.1 < x =< 0.3, Kết luận : Mức độ cảnh báo Trung bình. Giá trị mô hình có tính ổn định trung bình},
        Cảnh báo Cao : {Ngưỡng giá trị : 0.3 < x, Kết luận : Mức độ cảnh báo Cao. Giá trị mô hình có tính ổn định thấp},
    }""",   
    
    "AR_model" :
    """Bài kiểm thử: AR mô hình (AR_model),
    Đánh giá kết quả bài kiểm thử : {
        Cảnh báo Thấp : {Ngưỡng giá trị : 0.6 =< x, Kết luận : Mức độ cảnh báo Thấp. Mô hình có khả năng phân biệt tốt},
        Cảnh báo Trung bình : {Ngưỡng giá trị : 0.3 =< x < 0.6, Kết luận : Mức độ cảnh báo Trung bình. Mô hình có khả năng phân biệt trung bình},
        Cảnh báo Cao : {Ngưỡng giá trị : x < 0.3, Kết luận : Mức độ cảnh báo Cao. Mô hình có khả năng phân biệt thấp},
    }""",
    
    "AR_feature" :
    """Bài kiểm thử: AR dữ liệu (AR_feature),
    Đánh giá kết quả bài kiểm thử : {
        Cảnh báo Thấp : {Ngưỡng giá trị : 0.6 =< x, Kết luận : Mức độ cảnh báo Thấp. Biến có khả năng phân biệt tốt},
        Cảnh báo Trung bình : {Ngưỡng giá trị : 0.3 =< x < 0.6, Kết luận : Mức độ cảnh báo Trung bình. Biến có khả năng phân biệt trung bình},
        Cảnh báo Cao : {Ngưỡng giá trị : x < 0.3, Kết luận : Mức độ cảnh báo Cao. Biến có khả năng phân biệt thấp},
    }""",
          
}

# Tạo các câu lệnh query cho từng mô hình
all_query = []
segmentation_col = 'Phân khúc'
component_col = 'Cấu phần mô hình'
for test_case in df:
    df_by_testcase = df[test_case].ffill()
    for test_segment in df_by_testcase[segmentation_col].unique():
        df_by_segment = df_by_testcase[df_by_testcase[segmentation_col] == test_segment]
        for component in df_by_segment[component_col].unique():
            df_by_component = df_by_segment[df_by_segment[component_col] == component]
            test_result = df_by_component.T.to_json(force_ascii = False)
            query = Introduction + "Ngưỡng đánh giá: " + str(threshold[test_case]) + Command + "Kết quả mô hình: " + str(test_result)
            all_query.append([test_case, test_segment, component, query])

list_query = pd.DataFrame(all_query).reset_index(drop=True)
list_query.rename(columns={0: 'Test', 1: 'Model', 2: 'Component', 3: 'Query'}, inplace=True)

result_query = []
for i in stqdm(list_query.index):
    print(list_query['Test'][i] + list_query['Model'][i])
    text_ = ask(client, mess=list_query['Query'][i])
    result_query.append(text_)


# Tạo trường kết quả theo từng bài giám sát
list_query['Results'] = result_query

for i in df.keys():
    if i != 'AR_feature':
        df[i]['Results'] = list_query[list_query.Test == i].Results.values
    else:
        df['AR_feature'].loc[df['AR_feature']['Cấu phần mô hình'].notna(), 'Results'] = list_query[list_query.Test == 'AR_feature'].Results.values

# Query ChatGPT tổng hợp các bài giám sát theo mô hình
Model_intro = """
Sử dụng các thông tin được cung cấp dưới đây, hãy tổng hợp kết luận về hiệu năng và chất lượng của mô hình từ kết luận các bài kiểm thử.
"""

Model_assessment = """
Yêu cầu: Tổng hợp kết luận từ các bài kiểm thử theo nguyên tắc:
1. Đủ thông tin các bài kiểm thử được sử dụng.
2. Phải đưa ra được kết luận về hiệu năng mô hình.
3. Tóm tắt ngắn gọn trong 100 từ.
"""


# Loop tổng hợp kết quả các bài test theo model
model_test_result = []
for model in list_query['Model'].unique():
    for comp in list_query['Component'].unique():
        model_test = list_query[(list_query['Model'] == model) & (list_query['Component'] == comp)]
        all_test_results = (model_test.Test.str.strip() + ": " + model_test.Results.apply(lambda x: '\n'.join(x).replace('[', '').replace(']', '')).str.strip())
        model_test_result.append([model, comp, '\n'.join(all_test_results)])

# Tạo query cho từng mô hình
combine_tests_by_model = pd.DataFrame(model_test_result)
combine_tests_by_model.rename(columns={0: 'Model', 1: 'Component', 2: 'All_test_summary'}, inplace=True)
combine_tests_by_model['Summary_query'] = Model_intro + "\n Kết luận các bài kiểm thử: " + combine_tests_by_model['All_test_summary'] + Model_assessment

# Loop tổng hợp thông qua chatGPT
summary_result = []
for i in stqdm(combine_tests_by_model.index):
    summary_text = ask(client, combine_tests_by_model['Summary_query'][i])
    summary_result.append('\n'.join(summary_text))

combine_tests_by_model['Summary result'] = summary_result

df['Kết luận mô hình'] = combine_tests_by_model[['Model', 'Component', 'Summary result']].rename(columns={'Model': 'Mô hình', 'Component': 'Cấu phần mô hình', 'Summary result': 'Kết luận'})

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
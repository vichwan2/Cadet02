import pandas as pd
import json
import re
import os

# CSV 파일 경로
csv_file = '/root/eliceAI/mm_jh/mmdetection/results/results.csv'
output_csv_file = '/root/eliceAI/mm_jh/mmdetection/Submission/output_file.csv'
# 이미지 크기 (예시, 실제 이미지 크기에 맞게 조정)
image_width = 1280
image_height = 720


# CSV 파일 읽기
df = pd.read_csv(csv_file)

# 4번째 열과 8번째 열 선택 (열 인덱스는 0부터 시작하므로 3과 7)
df = df[['img_path','pred_instances']]

# 첫 번째 열에서 파일 경로의 파일 이름만 추출
df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: os.path.basename(x).split('.')[0])


print("CSV 파일이 성공적으로 저장되었습니다.")
def convert_tensor_to_list(tensor_str):
    """
    Convert tensor notation in string to a list notation.
    Example:
    - tensor([0.8276, 0.8031]) -> [0.8276, 0.8031]
    - tensor([[100, 150, 200, 250], [300, 350, 400, 450]]) -> [[100, 150, 200, 250], [300, 350, 400, 450]]
    """
    # Remove 'tensor(' and ')', convert to list format
    tensor_str = re.sub(r'tensor\(\s*\[\s*(.*?)\s*\]\s*\)', r'[\1]', tensor_str)  # For lists
    tensor_str = re.sub(r'tensor\(\s*(.*?)\s*\)', r'[\1]', tensor_str)  # For single values
    tensor_str = re.sub(r'tensor\(', '', tensor_str)  # Remove 'tensor('
    tensor_str = re.sub(r'\)', '', tensor_str)  # Remove ')'
    return tensor_str

# 'pred_instances' 열의 텐서 문자열을 변환
df['pred_instances'] = df['pred_instances'].apply(convert_tensor_to_list)

# 수정된 데이터프레임을 새로운 CSV 파일로 저장
df.to_csv(output_csv_file, index=False)

print(f"CSV 파일이 '{output_csv_file}'로 성공적으로 저장되었습니다.")
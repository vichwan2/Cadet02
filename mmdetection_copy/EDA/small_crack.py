import pandas as pd
import numpy as np
import json
import os
import shutil

# 설정
input_json_path = '/root/eliceAI/dataset_root/annotations/merged_train.json'  # COCO 포맷 JSON 파일 경로
output_folder = '/root/eliceAI/mm_jh/mmdetection/EDA/small_crack'  # 추출된 이미지를 저장할 폴더
max_sqrt_area = 15  # 기준 sqrt_area 값

# JSON 파일 로딩
with open(input_json_path, 'r') as f:
    coco_data = json.load(f)

# 바운딩 박스 정보 데이터프레임 생성
df = pd.json_normalize(coco_data['annotations'])
df[["X", "Y", "W", "H"]] = list(df.bbox)
df.drop(columns='bbox', inplace=True)
df['sqrt_area'] = np.sqrt(df['area'])

# 특정 sqrt_area 이하인 바운딩 박스가 있는 이미지 ID 추출
filtered_images_ids = df[df['sqrt_area'] <= max_sqrt_area]['image_id'].unique()

# category_id=2를 포함하는 이미지 ID를 찾기
excluded_image_ids = df[df['category_id'] == 2]['image_id'].unique()

# 최종 필터링된 이미지 ID
final_image_ids = [img_id for img_id in filtered_images_ids if img_id not in excluded_image_ids]

# 이미지 ID와 파일명 매핑
image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

# 결과를 저장할 폴더 생성 (존재하지 않으면 생성)
os.makedirs(output_folder, exist_ok=True)

# 이미지 추출
for image_id in final_image_ids:
    if image_id in image_id_to_filename:
        image_file = image_id_to_filename[image_id]
        # 이미지 파일 경로를 지정해야 합니다.
        # 여기서는 이미지가 현재 작업 디렉토리의 `images` 폴더에 있다고 가정합니다.
        image_path = os.path.join('/root/eliceAI/dataset_root/train', image_file)
        
        if os.path.exists(image_path):
            shutil.copy(image_path, output_folder)
        else:
            print(f"Image {image_path} does not exist.")

print(f"Extracted {len(final_image_ids)} images.")

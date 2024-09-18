from PIL import Image
from pathlib import Path
import json

# 훈련 데이터의 어노테이션 파일 경로 (카테고리 정보를 불러오기 위해 사용)
train_ann_file = '/root/eliceAI/dataset_root/annotations/train.json'

# 테스트 이미지 경로
test_images_path = "/root/eliceAI/dataset_root/test"
filenames = list(Path(test_images_path).glob("*.jpg"))  # .jpg 파일 리스트 추출

# 이미지 정보 추출 및 어노테이션 생성
images = []
for idx, filename in enumerate(filenames):
    with Image.open(filename) as img:
        width, height = img.size
        images.append({
            'file_name': filename.name,  # 경로를 제거하고 파일 이름만 남김
            'height': height,
            'width': width,
            'id': idx + 1
        })

# 훈련 데이터에서 카테고리 정보 불러오기
with open(train_ann_file, 'r') as f:
    train_ann = json.load(f)
categories = train_ann['categories']

# 결과를 json 파일로 저장할 때 카테고리 정보와 함께 포맷팅
output_file = '/root/eliceAI/dataset_root/annotations/test.json'

with open(output_file, 'w') as f:
    json.dump({"categories": categories, "images": images}, f, indent=4)  # 카테고리와 이미지 정보 추가

print(f"카테고리가 포함된 테스트 어노테이션 파일이 {output_file}에 생성되었습니다.")

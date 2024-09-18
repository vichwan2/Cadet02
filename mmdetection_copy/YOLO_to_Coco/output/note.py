import json

# JSON 파일 불러오기
try:
    with open('/root/eliceAI/Yolo-to-COCO-format-converter/output/train.json', 'r') as f:
        data = json.load(f)
    print("파일이 올바른 JSON 형식입니다.")
except json.JSONDecodeError:
    print("파일이 올바르지 않은 JSON 형식입니다.")

import pandas as pd
import json
import re

# CSV 파일 경로
csv_file = '/root/eliceAI/mm_jh/mmdetection/Submission/output_file.csv'

# 이미지 크기 (예시, 실제 이미지 크기에 맞게 조정)
image_width = 1280
image_height = 720

# CSV 파일 읽기
df = pd.read_csv(csv_file)

def convert_bbox_to_yolo(bbox, image_width, image_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return [x_center, y_center, width, height]

def process_row(row):
    # JSON 형식으로 변환할 때 작은따옴표를 큰따옴표로 변경
    pred_instances_str = row['pred_instances'].replace("'", "\"")
    
    try:
        pred_instances = json.loads(pred_instances_str)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        print(f"String causing the error: {pred_instances_str}")
        return None
    
    scores = pred_instances['scores']
    labels = pred_instances['labels']
    bboxes = pred_instances['bboxes']
    
    yolo_bboxes = [convert_bbox_to_yolo(bbox, image_width, image_height) for bbox in bboxes]
    
    # 각 bbox에 대해 YOLO 형식으로 변환
    detections = []
    for score, label, yolo_bbox in zip(scores, labels, yolo_bboxes):
        detection = {
            "class_id": int(label),
            "conf": float(score),
            "x": yolo_bbox[0],
            "y": yolo_bbox[1],
            "w": yolo_bbox[2],
            "h": yolo_bbox[3]
        }
        detections.append(detection)
    
    return json.dumps(detections)

# YOLO 형식으로 변환한 결과를 저장할 데이터프레임
new_df_list = []

# 각 행 처리
for index, row in df.iterrows():
    labels_string = process_row(row)
    if labels_string:
        new_df_list.append({'id': row['img_path'], 'labels': labels_string})

# 리스트를 DataFrame으로 변환
new_df = pd.DataFrame(new_df_list)

# 결과를 새로운 CSV 파일로 저장
output_file = '/root/eliceAI/mm_jh/mmdetection/Submission/submission.csv'
new_df.to_csv(output_file, index=False)

print("CSV 파일이 성공적으로 변환되었습니다.")
import pickle
import csv

# pkl 파일 경로와 csv 파일 경로
pkl_file_path = '/root/eliceAI/mm_jh/mmdetection/results/results.pkl'
csv_file_path = '/root/eliceAI/mm_jh/mmdetection/results/results.csv'


# pkl 파일 열기
with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)

# CSV 파일로 저장하기
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # 데이터가 리스트 또는 딕셔너리 형식이라면
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        header = data[0].keys()  # 딕셔너리의 키를 헤더로 사용
        writer.writerow(header)
        
        # 데이터 작성
        for row in data:
            writer.writerow(row.values())

print(f"PKL 파일이 {csv_file_path}로 변환되었습니다.")

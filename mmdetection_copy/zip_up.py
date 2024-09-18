import shutil
import os

# 압축할 폴더 경로와 생성할 zip 파일의 경로 설정
folder_path = '/root/eliceAI/Histo_datasets'  # 예: 'C:/Users/Username/Documents/myfolder'
zip_file_path = '/root/RDD2020'  # 예: 'C:/Users/Username/Documents/myfolder.zip'

# 폴더를 zip 파일로 압축
shutil.make_archive(zip_file_path.replace('.zip', ''), 'zip', folder_path)

print(f"{zip_file_path} 파일이 생성되었습니다.")

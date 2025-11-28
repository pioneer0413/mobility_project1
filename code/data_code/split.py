import os
import glob
import random

# =========================================================
# 1. 설정
# =========================================================

# 이미지가 있는 폴더 경로
IMAGE_DIR = './train/images'

# 저장할 텍스트 파일 이름
TRAIN_LIST_FILE = 'train.txt'
VAL_LIST_FILE = 'val.txt'

# 분할 비율 (Train : Val = 6 : 4)
SPLIT_RATIO = 0.6 

def split_dataset():
    # 1. 이미지 파일 목록 가져오기
    # 지원할 확장자 목록
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    # 2. 절대 경로로 변환 (YOLO 학습 시 경로 에러 방지)
    image_paths = [os.path.abspath(path) for path in image_paths]
    
    # 3. 랜덤 셔플 (중요: 순서를 섞어야 골고루 나뉨)
    random.shuffle(image_paths)
    
    # 4. 분할 지점 계산
    split_index = int(len(image_paths) * SPLIT_RATIO)
    
    train_images = image_paths[:split_index]
    val_images = image_paths[split_index:]
    
    print(f"총 이미지 개수: {len(image_paths)}")
    print(f"Train 개수 (60%): {len(train_images)}")
    print(f"Val 개수 (40%): {len(val_images)}")
    
    # 5. 파일로 저장
    with open(TRAIN_LIST_FILE, 'w') as f:
        f.write('\n'.join(train_images))
        
    with open(VAL_LIST_FILE, 'w') as f:
        f.write('\n'.join(val_images))
        
    print(f"\n[완료] '{TRAIN_LIST_FILE}'와 '{VAL_LIST_FILE}'이 생성되었습니다.")

if __name__ == '__main__':
    split_dataset()
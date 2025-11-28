import os
import cv2
import glob
from tqdm import tqdm

# =========================================================
# 1. 설정 (KITTI 표준 클래스 적용)
# =========================================================

# 경로 설정 (사용자 환경에 맞춤)
INPUT_LABEL_DIR = './train/train_label(original)'
OUTPUT_LABEL_DIR = './train/train_label'
IMAGE_DIR = './train/train_images'

# ★ KITTI 데이터셋의 표준 클래스 정의
# 이 리스트의 순서대로 YOLO의 클래스 ID (0, 1, 2...)가 결정됩니다.
KITTI_CLASSES = [
    'Car',              # ID: 0
    'Van',              # ID: 1
    'Truck',            # ID: 2
    'Pedestrian',       # ID: 3
    'Person_sitting',   # ID: 4
    'Cyclist',          # ID: 5
    'Tram'           # ID: 6
]

# 학습에서 제외할 클래스
IGNORE_CLASSES = ['DontCare','Misc']

# 클래스 이름 -> ID 매핑 딕셔너리 생성
CLASS_MAP = {name: i for i, name in enumerate(KITTI_CLASSES)}

# =========================================================
# 2. 변환 로직
# =========================================================

def convert_box(size, box):
    """ KITTI(x1, y1, x2, y2) -> YOLO(cx, cy, w, h) 변환 """
    dw = 1. / size[0]
    dh = 1. / size[1]
    
    # 중심점
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    
    # 너비, 높이
    w = box[2] - box[0]
    h = box[3] - box[1]
    
    # 정규화
    cx = cx * dw
    w = w * dw
    cy = cy * dh
    h = h * dh
    return (cx, cy, w, h)

def main():
    if not os.path.exists(OUTPUT_LABEL_DIR):
        os.makedirs(OUTPUT_LABEL_DIR)

    # 텍스트 파일 목록
    txt_files = glob.glob(os.path.join(INPUT_LABEL_DIR, '*.txt'))
    print(f"총 {len(txt_files)}개의 파일을 처리합니다.")

    converted_count = 0
    
    for txt_file in tqdm(txt_files):
        filename = os.path.basename(txt_file)
        file_id = os.path.splitext(filename)[0]
        
        # 1. 이미지 찾기 (이미지 크기가 필요함)
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            temp = os.path.join(IMAGE_DIR, file_id + ext)
            if os.path.exists(temp):
                image_path = temp
                break
        
        if image_path is None:
            # 이미지가 없으면 변환 불가 (Skip)
            continue

        # 2. 이미지 로드
        img = cv2.imread(image_path)
        if img is None: continue
        h, w, _ = img.shape

        # 3. 라벨 읽기 및 변환
        yolo_lines = []
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 8: continue
            
            obj_name = parts[0] # 예: Car, Truck...
            
            # 무시할 클래스면 패스
            if obj_name in IGNORE_CLASSES:
                continue
            
            # 리스트에 없는 엉뚱한 클래스면 경고 출력 후 패스
            if obj_name not in CLASS_MAP:
                # print(f"Warning: 알 수 없는 클래스 '{obj_name}' 발견. 제외합니다.")
                continue

            class_id = CLASS_MAP[obj_name]
            
            # KITTI bbox: index 4, 5, 6, 7
            try:
                x1 = float(parts[4])
                y1 = float(parts[5])
                x2 = float(parts[6])
                y2 = float(parts[7])
            except ValueError:
                continue

            # YOLO 포맷으로 변환
            bb = convert_box((w, h), (x1, y1, x2, y2))
            
            yolo_lines.append(f"{class_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")

        # 4. 저장 (변환된 내용이 있을 때만)
        if yolo_lines:
            save_path = os.path.join(OUTPUT_LABEL_DIR, filename)
            with open(save_path, 'w') as f_out:
                f_out.write('\n'.join(yolo_lines))
            converted_count += 1

    print(f"\n[완료] 총 {converted_count}개의 파일이 변환되어 저장되었습니다.")
    print(f"저장 경로: {OUTPUT_LABEL_DIR}")
    print("------------------------------------------------")
    print("★ 중요: 아래 내용을 복사해서 새로운 data.yaml을 만드세요!")
    print(f"names: {KITTI_CLASSES}")
    print(f"nc: {len(KITTI_CLASSES)}")

if __name__ == '__main__':
    main()
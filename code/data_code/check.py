import matplotlib
# 서버 환경에서 에러 없이 이미지를 저장하기 위해 백엔드 설정 (반드시 import pyplot 이전에 해야 함)
matplotlib.use('Agg') 
import cv2
import matplotlib.pyplot as plt
import os
import glob
import random

# =========================================================
# 1. 설정
# =========================================================

IMAGE_DIR = './train/train_images'
LABEL_DIR = './train/train_label'
CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
NUM_SAMPLES = 4  # 확인할 이미지 개수

def save_yolo_labels():
    label_files = glob.glob(os.path.join(LABEL_DIR, '*.txt'))
    
    if not label_files:
        print("라벨 파일이 없습니다.")
        return

    samples = random.sample(label_files, min(len(label_files), NUM_SAMPLES))
    
    # 그래프 크기 설정
    plt.figure(figsize=(15, 5 * len(samples)))

    for i, label_path in enumerate(samples):
        filename = os.path.basename(label_path)
        file_id = os.path.splitext(filename)[0]
        
        # 이미지 찾기
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            temp = os.path.join(IMAGE_DIR, file_id + ext)
            if os.path.exists(temp):
                img_path = temp
                break
        
        if img_path is None: continue

        img = cv2.imread(img_path)
        if img is None: continue
        
        h, w, _ = img.shape
        
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            
            # YOLO 좌표 역변환
            cx, cy, bw, bh = map(float, parts[1:])
            
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            # 박스 그리기
            color = (0, 255, 0)
            label_name = CLASSES[class_id] if class_id < len(CLASSES) else str(class_id)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Plot에 추가
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(len(samples), 1, i + 1)
        plt.imshow(img_rgb)
        plt.title(f"{filename}")
        plt.axis('off')

    # ★ 화면 출력 대신 파일로 저장
    save_filename = 'check_result.png'
    plt.tight_layout()
    plt.savefig(save_filename)
    print(f"\n[완료] 결과 이미지가 '{save_filename}' 파일로 저장되었습니다.")
    print("왼쪽 파일 탐색기에서 해당 이미지를 클릭해서 확인해보세요.")

if __name__ == '__main__':
    save_yolo_labels()
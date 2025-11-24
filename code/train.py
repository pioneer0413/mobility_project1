import os
from datetime import datetime
from ultralytics import YOLO

# =========================================================
# ✅ [사용자 설정] 
# =========================================================

USER_NAME = "kwkoo" # ★ 여기에 본인 이름(또는 프로젝트명)을 적으세요!
MODEL_NUM = 3       # 1(Nano), 2(Small), 3(Medium), 4(Large), 5(XLarge)
EPOCHS = 50         
BATCH_SIZE = 64     
GPUS = [0, 1, 2, 3] 

# =========================================================
# 1. 모델 딕셔너리
# =========================================================
MODEL_DICT = {
    1: 'yolo11n.pt',
    2: 'yolo11s.pt',
    3: 'yolo11m.pt',
    4: 'yolo11l.pt',
    5: 'yolo11x.pt'
}

def main():
    # -----------------------------------------------------
    # 2. 폴더 이름 생성 로직 (이름 + 모델 + 시간)
    # -----------------------------------------------------
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    if MODEL_NUM in MODEL_DICT:
        model_name = MODEL_DICT[MODEL_NUM]
    else:
        model_name = MODEL_DICT[1]
    
    pure_model_name = model_name.replace('.pt', '')

    # ★ 최종 폴더 이름 형식: kwkoo_yolo11m_20251124_140000
    # 만약 "train_"을 앞에 붙이고 싶다면 f"train_{USER_NAME}_..." 로 수정하세요.
    folder_name = f"{USER_NAME}_{pure_model_name}_{current_time}"

    print(f"\n" + "="*40)
    print(f"▶ [INFO] 사용자: {USER_NAME}")
    print(f"▶ [INFO] 저장 폴더명: {folder_name}")
    print("="*40 + "\n")

    # -----------------------------------------------------
    # 3. 경로 설정
    # -----------------------------------------------------
    data_config = os.path.abspath("../data/data.yaml") # 소문자 data 주의
    project_path = os.path.abspath("../model")

    # -----------------------------------------------------
    # 4. 학습 시작
    # -----------------------------------------------------
    model = YOLO(model_name)

    results = model.train(
        data=data_config,
        epochs=EPOCHS,
        imgsz=640,
        device=GPUS,
        batch=BATCH_SIZE,
        workers=16,
        project=project_path,
        
        name=folder_name, # ★ 생성한 폴더 이름 적용
        
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True
    )

if __name__ == '__main__':
    main()
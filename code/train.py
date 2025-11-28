import os
from datetime import datetime
from ultralytics import YOLO

# =========================================================
# ✅ [사용자 설정]
# =========================================================

USER_NAME = "kwkoo" 
MODEL_NUM = 3       # yolo11m.pt
GPUS = [0, 1, 2, 3] # 요청하신 대로 4개 사용
BATCH_SIZE = 96     # args.yaml: 64

EPOCHS = 100        # args.yaml: 50 (기존 100에서 변경)
PATIENCE = 100      # args.yaml: 100 (기존 10에서 변경)
WORKERS = 16        # args.yaml: 2 (기존 16에서 변경)
SEED = 0            # args.yaml: 0

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
    # 2. 폴더 이름 생성
    # -----------------------------------------------------
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    if MODEL_NUM in MODEL_DICT:
        model_name = MODEL_DICT[MODEL_NUM]
    else:
        model_name = MODEL_DICT[1]
    
    pure_model_name = model_name.replace('.pt', '')
    folder_name = f"{USER_NAME}_{pure_model_name}_{current_time}"

    print(f"\n" + "="*40)
    print(f"▶ [INFO] 사용자: {USER_NAME}")
    print(f"▶ [INFO] 모델: {model_name}")
    print(f"▶ [INFO] GPU 설정: {GPUS}")
    print(f"▶ [INFO] 최대 에포크: {EPOCHS}")
    print(f"▶ [INFO] 얼리스탑(Patience): {PATIENCE}")
    print(f"▶ [INFO] 저장 폴더: {folder_name}")
    print("="*40 + "\n")

    # -----------------------------------------------------
    # 3. 경로 설정
    # -----------------------------------------------------
    # 경로가 올바른지 확인해주세요. (상대경로 vs 절대경로)
    data_config = os.path.abspath("../data/data.yaml")
    project_path = os.path.abspath("../model")

    # -----------------------------------------------------
    # 4. 학습 시작
    # -----------------------------------------------------
    model = YOLO(model_name)

    results = model.train(
        # --- 기본 시스템/환경 설정 ---
        data=data_config,
        epochs=EPOCHS,     
        patience=PATIENCE, 
        batch=BATCH_SIZE,
        imgsz=640,
        device=GPUS,        # 요청: GPU 4개 사용
        workers=WORKERS,
        project=project_path,
        name=folder_name,
        
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=SEED,
        deterministic=True,
        save=True,
        
        # --- ✅ Best Hyperparameters 적용 ---
        optimizer='SGD',
        lr0=0.0023212750181555836,
        weight_decay=4.8606556162235835e-05,
        momentum=0.9169929265813977,
        warmup_epochs=1,

        # --- ⚠️ 나머지 하이퍼파라미터(Augmentation, Loss gain 등)는 삭제함 ---
        # 삭제된 항목들은 ultralytics 내부의 default.yaml 값을 자동으로 따르게 됩니다.
    )

if __name__ == '__main__':
    main()
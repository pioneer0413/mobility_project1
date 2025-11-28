import os
from datetime import datetime
from ultralytics import YOLO

# =========================================================
# ✅ [사용자 설정] 환경 및 튜닝 모드 설정
# =========================================================

USER_NAME = "kwkoo" 
MODEL_NUM = 3       # 3 = yolo11m.pt
GPUS = [0, 1, 2, 3] # 4개 GPU 모두 사용
BATCH_SIZE = 96     # 📉 [안정성] OOM 방지를 위해 32 -> 16으로 감소

# 🚀 [초고속 튜닝 설정]
# 전체 데이터를 다 보면 너무 느리므로, 10%만 샘플링하여 경향성을 파악합니다.
DATA_FRACTION = 0.1     # 데이터의 10%만 사용 
EPOCHS_PER_TRIAL = 20   # 10 에포크면 학습률의 좋고 나쁨을 판별하기 충분함
TUNE_ITERATIONS = 15    # 핵심 파라미터만 찾으므로 15회 반복이면 적당함

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

# =========================================================
# 🎯 [핵심] 튜닝 검색 공간 (Search Space)
# =========================================================
# AI가 이 범위 안에서만 값을 찾도록 제한합니다.
# 불필요한 Augmentation 탐색은 제외하고, 학습 성능에 직결된 값만 찾습니다.
def kitti_search_space(trial=None):
    return {
        # 1. 학습 엔진 (가장 중요)
        'lr0': (1e-5, 1e-2),          # 초기 학습률
        'lrf': (0.01, 1.0),           # 최종 학습률 비율
        'momentum': (0.6, 0.98),      # 모멘텀 (SGD 안정성)
        'weight_decay': (0.0001, 0.001), # 가중치 감쇠 (과적합 방지)
        
        # 2. 손실 가중치 (정확도 조절)
        'box': (0.05, 10.0),          # 박스 정확도 중요도
        'cls': (0.2, 4.0),            # 클래스 분류 중요도
        
        # 나머지는 탐색하지 않고 기본값 사용
    }

def main():
    # -----------------------------------------------------
    # 2. 폴더 이름 및 정보 출력
    # -----------------------------------------------------
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    if MODEL_NUM in MODEL_DICT:
        model_name = MODEL_DICT[MODEL_NUM]
    else:
        model_name = MODEL_DICT[1]
    
    pure_model_name = model_name.replace('.pt', '')
    folder_name = f"{USER_NAME}_{pure_model_name}_FAST_TUNE_{current_time}"

    print(f"\n" + "="*50)
    print(f"▶ [INFO] 모드: KITTI 초고속 튜닝 (Fraction {DATA_FRACTION*100}%)")
    print(f"▶ [INFO] 모델: {model_name}")
    print(f"▶ [INFO] 해상도: 1280 (고해상도 유지)")
    print(f"▶ [INFO] 배치: {BATCH_SIZE} (GPU 메모리 최적화)")
    print(f"▶ [INFO] 저장 폴더: {folder_name}")
    print("="*50 + "\n")

    # -----------------------------------------------------
    # 3. 경로 설정
    # -----------------------------------------------------
    data_config = os.path.abspath("../data/data.yaml")
    project_path = os.path.abspath("../model")

    # -----------------------------------------------------
    # 4. 튜닝 시작
    # -----------------------------------------------------
    model = YOLO(model_name)

    model.tune(
        data=data_config,
        
        # 튜닝 반복 설정
        epochs=EPOCHS_PER_TRIAL,  
        iterations=TUNE_ITERATIONS,
        
        # ✅ [중요] 정의한 검색 공간 함수 실행 결과를 전달
        space=kitti_search_space(), 
        
        # 하드웨어 설정
        device=GPUS,
        batch=BATCH_SIZE,
        workers=16,       # 메모리가 부족하면 2로 줄이세요
        project=project_path,
        name=folder_name,
        exist_ok=True,
        
        # ✅ [속도 및 성능 최적화 설정]
        imgsz=640,      # 해상도 유지 (작은 객체 탐지 필수)
        fraction=DATA_FRACTION, # 🔥 핵심: 데이터 20%만 사용 (속도 UP)
        rect=True,       # 직사각형 학습 (KITTI 비율 최적화)
        
        optimizer='auto', # 최적화기 자동 선택
        
        # ✅ [고정 파라미터] 물리적 특성 반영 (탐색 제외)
        degrees=0.0,     # 회전 끄기
        flipud=0.0,      # 상하반전 끄기
        
        # 기타 설정
        val=True,       
        plots=False,     
        save=False       
    )

    print(f"\n✅ 튜닝 완료! {project_path}/{folder_name}/best_hyperparameters.yaml 파일을 확인하세요.")

if __name__ == '__main__':
    main()
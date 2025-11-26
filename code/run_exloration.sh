# !/bin/bash

# 모델 경로 리스트
MODEL_PATHS=(
    "/home/itec/mobility_project1/model/enric_yolo11l_20251124_150213/weights/best.engine"
    "/home/itec/mobility_project1/model/enric_yolo11m_20251124_105625/weights/best.pt"
)

# exploration.py 실행
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    python3 code/exploration.py --model_path "$MODEL_PATH" --mode "full"
done
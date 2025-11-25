# 프로그램 관리
import os
import argparse 
from datetime import datetime
import json
# 모델
from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO TensorRT Inference Script")
    # 사용자명
    parser.add_argument('--user', type=str, default='hwkang', 
                        help='User name for saving results')
    # 실행 모드 ['engine', 'predict']
    parser.add_argument('--mode', type=str, default='predict', choices=['build', 'predict'], 
                        help='Execution mode: build engine or predict')
    # 모델 경로
    parser.add_argument('--model_path', type=str, default='/home/hwkang/mobility_project1/model/kwkoo_yolo11m_20251124_142242/weights/best.pt', 
                        help='Path to the original YOLO model')
    # 파라미터 설정 파일 경로 configuration file path
    parser.add_argument('--config_path', type=str, default=None, 
                        help='Path to the configuration file (if any)')
    # 추론용 데이터 세트 경로
    parser.add_argument('--data_path', type=str, default='/home/hwkang/mobility_project1/data/test/image_2', 
                        help='Path to the dataset for inference')
    # 결과 저장 경로: default='./exp'
    parser.add_argument('--save_root_dir', type=str, default='./exp', 
                        help='Directory to save results')
    # 결과 저장 시 디렉터리 접미사
    parser.add_argument('--suffix', type=str, default=None, 
                        help='Suffix for the result directory name')
    args = parser.parse_args()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 설정 파일 로드 (필요한 경우)
    if args.config_path is not None:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        # 설정 파일에서 필요한 매개변수를 로드하여 args에 적용할 수 있음
        # 예: args.some_param = config['some_param']
    else:
        config = {}

    # 엔진 빌드 또는 추론
    # 빌드
    if args.mode == 'build':
        print("[INFO] Building TensorRT engine...")
        model = YOLO(args.model_path).export(
            format="engine",
            device=config.get('device', 0),
            half=config.get('half', True),
            imgsz=config.get('imgsz', 640),            # 고정 해상도
            dynamic=config.get('dynamic', False),      # 동적 shape 비활성화 → 최적화↑
            workspace=config.get('workspace', 4),      # TRT 빌드시 메모리 (GB)
        )
    # 추론
    elif args.mode == 'predict':
        print("[INFO] Running inference with TensorRT engine...")
        if os.path.exists(args.model_path):
            model = YOLO(args.model_path)
        else:
            raise FileNotFoundError(f"Engine file not found at {args.model_path}. Please build the engine first.")

        # 테스트 데이터 세트 경로 설정
        test_dir = args.data_path

        # 결과 저장 경로 설정
        save_root_dir = args.save_root_dir
        
        # `model_path`로 부터 모델 이름 추출
        model_name = args.model_path.split('/')[-3].split('_')[1] # ex) yolo11m

        save_dir_name = f"{args.user}_{model_name}_{current_time}"

        if args.suffix is not None:
            save_dir_name += f"_{args.suffix}"

        save_dir = os.path.join(save_root_dir, save_dir_name)

        # 추론 실행
        results = model.predict(
            source=test_dir,
            save=config.get('save', True),
            save_txt=config.get('save_txt', True),
            save_conf=config.get('save_conf', True),
            project=save_root_dir,
            name=save_dir_name,
            exist_ok=config.get('exist_ok', True),
            device=config.get('device', 0),
            half=config.get('half', True),
            imgsz=config.get('imgsz', 640),
        )

        print(f"[INFO] Inference completed. Results saved to {save_dir}")
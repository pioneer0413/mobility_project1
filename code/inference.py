# 프로그램 관리
import os
import argparse 
from datetime import datetime
import json
# 모델
from ultralytics import YOLO

'''
# 이미지 정보
result.orig_img        # 원본 이미지 (numpy array)
result.path            # 이미지 파일 경로
result.names           # 클래스 이름 딕셔너리 {0: 'person', 1: 'car', ...}

# 탐지 결과
result.boxes           # Boxes 객체 (바운딩 박스 정보)
result.boxes.xyxy      # 좌표 [x1, y1, x2, y2] (tensor)
result.boxes.xywh      # 좌표 [x_center, y_center, width, height]
result.boxes.conf      # 신뢰도 점수 (tensor)
result.boxes.cls       # 클래스 ID (tensor)

# 시간 정보 (딕셔너리)
result.speed           # {'preprocess': float, 'inference': float, 'postprocess': float}

# 저장/시각화
result.save(filename='result.jpg')  # 결과 이미지 저장
result.show()                        # 결과 표시
result.plot()
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO TensorRT Inference Script")
    # 사용자명
    parser.add_argument('--user', type=str, default='hwkang', 
                        help='User name for saving results')
    # 실행 모드 ['engine', 'predict']
    parser.add_argument('--mode', type=str, default='predict', choices=['build', 'predict'], 
                        help='Execution mode: build engine or predict')
    # 모델 경로
    parser.add_argument('--model_path', type=str, default='/home/hwkang/mobility_project1/model/enric_yolo11m_20251124_105625/weights/best.pt', 
                        help='Path to the original YOLO model')
    # 파라미터 설정 파일 경로 configuration file path
    parser.add_argument('--config_path', type=str, default=None, 
                        help='Path to the configuration file (if any)')
    # 추론용 데이터 세트 경로
    parser.add_argument('--data_path', type=str, default='/home/hwkang/mobility_project1/data/test/image_2_mini', 
                        help='Path to the dataset for inference')
    # 결과 저장 경로: default='./exp'
    parser.add_argument('--save_root_dir', type=str, default='./exp', 
                        help='Directory to save results')
    # 결과 저장 시 디렉터리 접미사
    parser.add_argument('--suffix', type=str, default=None, 
                        help='Suffix for the result directory name')
    # 실행 모드 ['single', 'iterative']
    parser.add_argument('--run_mode', type=str, default='iterative', choices=['single', 'iterative'], 
                        help='Run mode: single or iterative inference')
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
            half=config.get('half', True),              # FP16 사용 (속도↑, 정확도 약간↓)
            imgsz=config.get('imgsz', 640),             # 입력 이미지 크기
            dynamic=config.get('dynamic', False),       # 동적 shape (False 권장)
            workspace=config.get('workspace', 4),       # TRT 작업 메모리 (GB)
            batch=config.get('batch', 1),               # 배치 크기
            int8=config.get('int8', False),             # INT8 양자화 (속도↑↑, 정확도↓)
            simplify=config.get('simplify', True),      # ONNX 모델 단순화
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

        # 실행 모드에 따라 분기
        if args.run_mode == 'single':
            print("[INFO] Single inference mode selected.")
            # 단일 이미지 추론 (예: 첫 번째 이미지)
            image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(image_files) == 0:
                raise ValueError(f"No image files found in {test_dir}")
            
            for img_file in image_files:
                print(f"[INFO] Found image file: {img_file}")
                image_path = os.path.join(test_dir, img_file)
                results = model.predict(
                    source=image_path,
                    #save=config.get('save', True),
                    save_txt=config.get('save_txt', True),
                    #save_conf=config.get('save_conf', True),
                    project=save_root_dir,
                    name=save_dir_name,
                    exist_ok=config.get('exist_ok', True),
                    device=config.get('device', 0),
                    half=config.get('half', True),
                    imgsz=config.get('imgsz', 640),
                )
                result = results[0]
                
            print(f"[INFO] Inference completed on single image. Results saved to {save_dir}")
        elif args.run_mode == 'iterative':
            print("[INFO] Iterative inference mode selected.")
            
            # stream 사용 여부 확인
            use_stream = config.get('stream', False)
            
            # 추론 실행
            results = model.predict(
                source=test_dir,
                
                # 기본 설정
                device=config.get('device', 0),              # 정수로 수정
                half=config.get('half', True),
                imgsz=config.get('imgsz', 640),
                
                # 성능 최적화
                batch=config.get('batch', 8),
                stream=use_stream,                           # 제너레이터 반환 (메모리 효율)
                verbose=config.get('verbose', False),
                
                # NMS (후처리) 최적화
                conf=config.get('conf', 0.25),
                iou=config.get('iou', 0.7),
                max_det=config.get('max_det', 300),
                
                # 클래스 필터링
                classes=config.get('classes', None),
                
                # 이미지 전처리
                augment=config.get('augment', False),
                agnostic_nms=config.get('agnostic_nms', False),
                
                # 저장 옵션
                save_txt=config.get('save_txt', True),
                save_conf=config.get('save_conf', False),
                save_crop=config.get('save_crop', False),
                
                project=save_root_dir,
                name=save_dir_name,
                exist_ok=config.get('exist_ok', True),
            )

            # stream 모드에 따라 다르게 처리
            if use_stream:
                # 제너레이터 방식 - 메모리 효율적
                total_preprocess = 0
                total_inference = 0
                total_postprocess = 0
                num_images = 0
                
                for result in results:
                    total_preprocess += result.speed['preprocess']
                    total_inference += result.speed['inference']
                    total_postprocess += result.speed['postprocess']
                    num_images += 1
                    
                    if num_images % 100 == 0:
                        print(f"[INFO] Processed {num_images} images...")
                
                total_time = total_preprocess + total_inference + total_postprocess
                avg_inference = total_time / num_images if num_images > 0 else 0
                
            else:
                # 리스트 방식 - 한 번에 처리
                total_preprocess = sum(r.speed['preprocess'] for r in results)
                total_inference = sum(r.speed['inference'] for r in results)
                total_postprocess = sum(r.speed['postprocess'] for r in results)
                total_time = total_preprocess + total_inference + total_postprocess
                num_images = len(results)
                avg_inference = total_time / num_images if num_images > 0 else 0

            print(f"\n[INFO] Inference Statistics:")
            print(f"  Total images: {num_images}")
            print(f"  Average preprocess: {total_preprocess/num_images:.2f} ms")
            print(f"  Average inference: {total_inference/num_images:.2f} ms")
            print(f"  Average postprocess: {total_postprocess/num_images:.2f} ms")
            print(f"  Average total per image: {avg_inference:.2f} ms")
            print(f"  FPS: {1000/avg_inference:.2f}")

            print(f"[INFO] Inference completed. Results saved to {save_dir}")
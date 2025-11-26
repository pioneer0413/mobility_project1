# 프로그램 관리
import os
import argparse 
from datetime import datetime
import json
import csv
import cv2
import numpy as np
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
    parser.add_argument('--model_path', type=str, default='/home/itec/mobility_project1/model/enric_yolo11l_20251124_150213/weights/best.pt', 
                        help='Path to the original YOLO model')
    # 파라미터 설정 파일 경로 configuration file path
    parser.add_argument('--config_path', type=str, default=None, 
                        help='Path to the configuration file (if any)')
    # 추론용 데이터 세트 경로
    parser.add_argument('--data_path', type=str, default='/home/itec/mobility_project1/data/test/image_2_mini', 
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
    # CSV 저장 경로
    parser.add_argument('--csv_path', type=str, default='exp/speed.csv',
                        help='Path to save inference statistics CSV file')
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
        test_path = args.data_path
        
        # 입력이 비디오 파일인지 이미지 디렉터리인지 확인
        is_video = test_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

        # 결과 저장 경로 설정
        save_root_dir = args.save_root_dir
        # `model_path`로 부터 모델 이름 추출
        model_name = args.model_path.split('/')[-3].split('_')[1] # ex) yolo11m
        save_dir_name = f"{args.user}_{model_name}_{current_time}"
        if args.suffix is not None:
            save_dir_name += f"_{args.suffix}"
        save_dir = os.path.join(save_root_dir, save_dir_name)

        # 통계 저장용 변수 초기화
        # 성능 관련 통계는 나중에 추가 (앞쪽에 배치하기 위해)
        stats = {}
        
        # config에서 실제로 사용된 값들을 stats에 추가 (default 값 포함)
        # 각 predict 호출에서 실제로 사용되는 파라미터들
        inference_params = {
            'device': config.get('device', 0),
            'half': config.get('half', True),
            'imgsz': config.get('imgsz', 1280),
            'conf': config.get('conf', 0.25),
            'iou': config.get('iou', 0.7),
            'max_det': config.get('max_det', 300),
            'batch': config.get('batch', 1),
            'stream': config.get('stream', False),
            'verbose': config.get('verbose', False),
            'augment': config.get('augment', False),
            'agnostic_nms': config.get('agnostic_nms', False),
            'save_txt': config.get('save_txt', False),
            'save_conf': config.get('save_conf', False),
            'save_crop': config.get('save_crop', False),
            'exist_ok': config.get('exist_ok', True),
            'show_video': config.get('show_video', False),
            'save_video': config.get('save_video', False),
            'video_fps': config.get('video_fps', 10),
        }
        
        # classes는 None일 수 있으므로 별도 처리
        classes = config.get('classes', None)
        if classes is not None:
            inference_params['classes'] = str(classes)  # 리스트를 문자열로 변환
        
        # 메타 정보 (config 뒤에 배치)
        meta_info = {
            'timestamp': current_time,
            'user': args.user,
            'model_name': model_name,
            'model_path': args.model_path,
            'data_path': args.data_path,
            'run_mode': args.run_mode,
            'is_video': is_video,
        }

        # 실행 모드에 따라 분기
        if args.run_mode == 'single':
            print("[INFO] Single inference mode selected - Real-time frame-by-frame processing.")
            
            if is_video:
                # 비디오 파일 열기
                cap = cv2.VideoCapture(test_path)
                if not cap.isOpened():
                    raise ValueError(f"Failed to open video file: {test_path}")
                
                # 비디오 정보 가져오기
                original_fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                print(f"[INFO] Video info: {width}x{height}, {original_fps:.2f} FPS, {total_frames} frames")
                
                # 시각화 옵션
                show_video = config.get('show_video', True)
                save_video = config.get('save_video', False)
                
                # 비디오 저장 설정
                video_writer = None
                if save_video:
                    video_path = os.path.join(save_root_dir, save_dir_name, 'inference_video.mp4')
                    os.makedirs(os.path.dirname(video_path), exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = config.get('video_fps', original_fps)
                    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                
                # 통계 변수
                total_preprocess = 0
                total_inference = 0
                total_postprocess = 0
                frame_count = 0
                
                # 프레임별 처리
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # 단일 프레임 추론 (즉시 처리)
                    results = model.predict(
                        source=frame,
                        device=config.get('device', 0),
                        half=config.get('half', True),
                        imgsz=config.get('imgsz', 640),
                        conf=config.get('conf', 0.25),
                        iou=config.get('iou', 0.7),
                        max_det=config.get('max_det', 300),
                        classes=config.get('classes', None),
                        verbose=False,
                        stream=False,  # single 모드는 stream 사용 안 함
                    )
                    
                    result = results[0]
                    
                    # 통계 수집
                    total_preprocess += result.speed['preprocess']
                    total_inference += result.speed['inference']
                    total_postprocess += result.speed['postprocess']
                    
                    # 바운딩 박스가 그려진 이미지 가져오기
                    annotated_img = result.plot()
                    
                    # FPS 계산 및 표시
                    current_time = result.speed['preprocess'] + result.speed['inference'] + result.speed['postprocess']
                    current_fps = 1000 / current_time if current_time > 0 else 0
                    
                    # 좌측 상단에 FPS 텍스트 추가
                    fps_text = f"FPS: {current_fps:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    font_thickness = 2
                    text_color = (0, 255, 0)
                    bg_color = (0, 0, 0)
                    
                    # 텍스트 크기 계산
                    (text_width, text_height), baseline = cv2.getTextSize(fps_text, font, font_scale, font_thickness)
                    
                    # 배경 사각형 및 텍스트 그리기
                    cv2.rectangle(annotated_img, (10, 10), (20 + text_width, 20 + text_height + baseline), bg_color, -1)
                    cv2.putText(annotated_img, fps_text, (15, 15 + text_height), font, font_scale, text_color, font_thickness)
                    
                    # 실시간 표시
                    if show_video:
                        cv2.imshow('YOLO Inference - Real-time', annotated_img)
                        
                        # 재생 속도 조절
                        wait_time = int(1000 / original_fps) if original_fps > 0 else 33
                        key = cv2.waitKey(wait_time) & 0xFF
                        
                        if key == ord('q'):
                            print("[INFO] User interrupted visualization")
                            break
                        elif key == ord('p'):  # 일시정지
                            print("[INFO] Paused. Press any key to continue...")
                            cv2.waitKey(0)
                    
                    # 비디오 파일에 저장
                    if save_video and video_writer is not None:
                        video_writer.write(annotated_img)
                    
                    # 진행 상황 로그
                    if frame_count % 30 == 0:
                        print(f"[INFO] Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
                
                # 리소스 정리
                cap.release()
                if show_video:
                    cv2.destroyAllWindows()
                if save_video and video_writer is not None:
                    video_writer.release()
                    print(f"[INFO] Video saved to {video_path}")
                
                # 통계 출력
                avg_preprocess = total_preprocess / frame_count if frame_count > 0 else 0
                avg_inference = total_inference / frame_count if frame_count > 0 else 0
                avg_postprocess = total_postprocess / frame_count if frame_count > 0 else 0
                avg_total = avg_preprocess + avg_inference + avg_postprocess
                fps = 1000 / avg_total if avg_total > 0 else 0
                
                print(f"\n[INFO] Single-frame Inference Statistics:")
                print(f"  Total frames: {frame_count}")
                print(f"  Average preprocess: {avg_preprocess:.2f} ms")
                print(f"  Average inference: {avg_inference:.2f} ms")
                print(f"  Average postprocess: {avg_postprocess:.2f} ms")
                print(f"  Average total per frame: {avg_total:.2f} ms")
                print(f"  Average FPS: {fps:.2f}")
                
                # 통계 저장
                # 순서: 성능 관련 -> config 파라미터 -> 메타 정보
                stats['total_frames'] = frame_count
                stats['avg_preprocess'] = avg_preprocess
                stats['avg_inference'] = avg_inference
                stats['avg_postprocess'] = avg_postprocess
                stats['fps'] = fps
                stats.update(inference_params)
                stats.update(meta_info)
                
            else:
                # 이미지 디렉터리인 경우 첫 번째 이미지만 처리
                image_files = [f for f in os.listdir(test_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if len(image_files) == 0:
                    raise ValueError(f"No image files found in {test_path}")
                
                img_file = image_files[0]
                print(f"[INFO] Found image file: {img_file}")
                image_path = os.path.join(test_path, img_file)
                
                results = model.predict(
                    source=image_path,
                    save_txt=config.get('save_txt', False),
                    project=save_root_dir,
                    name=save_dir_name,
                    exist_ok=config.get('exist_ok', True),
                    device=config.get('device', 0),
                    half=config.get('half', True),
                    imgsz=config.get('imgsz', 640),
                )
                result = results[0]
                
                print(f"[INFO] Inference completed on single image. Results saved to {save_dir}")
                
                # 단일 이미지는 통계 저장 안 함 (필요 시 추가 가능)
                stats = None
            
        elif args.run_mode == 'iterative':
            print("[INFO] Iterative inference mode selected - Batch processing then visualization.")
            
            # stream 사용 여부 확인
            use_stream = config.get('stream', False)
            
            # 시각화 옵션 추가
            show_video = config.get('show_video', False)  # 실시간 영상 표시 여부
            save_video = config.get('save_video', False)  # 영상 파일 저장 여부
            
            # 비디오 라이터 초기화 (저장 옵션이 켜져있을 경우)
            video_writer = None
            if save_video:
                video_path = os.path.join(save_root_dir, save_dir_name, 'inference_video.mp4')
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                
                # 해상도 가져오기
                if is_video:
                    # 비디오에서 해상도 가져오기
                    cap = cv2.VideoCapture(test_path)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                else:
                    # 첫 이미지로부터 해상도 가져오기
                    image_files = sorted([f for f in os.listdir(test_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    if image_files:
                        first_img = cv2.imread(os.path.join(test_path, image_files[0]))
                        height, width = first_img.shape[:2]
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = config.get('video_fps', 10)  # 기본 10 FPS
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            # 추론 실행
            results = model.predict(
                source=test_path,  # 비디오 파일 또는 이미지 디렉터리 모두 지원
                
                # 기본 설정
                device=config.get('device', 0),
                half=config.get('half', True),
                imgsz=config.get('imgsz', 640),
                
                # 성능 최적화
                batch=config.get('batch', 1) if not is_video else 1,  # 비디오는 배치 1
                stream=use_stream,
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
                save_txt=config.get('save_txt', False),
                save_conf=config.get('save_conf', False),
                save_crop=config.get('save_crop', False),
                
                project=save_root_dir,
                name=save_dir_name,
                exist_ok=config.get('exist_ok', True),
            )

            # stream 모드에 따라 다르게 처리
            total_preprocess = 0
            total_inference = 0
            total_postprocess = 0
            num_images = 0
            
            for result in results:
                total_preprocess += result.speed['preprocess']
                total_inference += result.speed['inference']
                total_postprocess += result.speed['postprocess']
                num_images += 1
                
                # 시각화 처리
                if show_video or save_video:
                    # 바운딩 박스가 그려진 이미지 가져오기
                    annotated_img = result.plot()  # BGR 이미지 반환
                    
                    # FPS 계산 및 표시
                    current_time = result.speed['preprocess'] + result.speed['inference'] + result.speed['postprocess']
                    current_fps = 1000 / current_time if current_time > 0 else 0
                    
                    # 좌측 상단에 FPS 텍스트 추가
                    fps_text = f"FPS: {current_fps:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    font_thickness = 2
                    text_color = (0, 255, 0)  # 녹색
                    bg_color = (0, 0, 0)  # 검은색 배경
                    
                    # 텍스트 크기 계산
                    (text_width, text_height), baseline = cv2.getTextSize(fps_text, font, font_scale, font_thickness)
                    
                    # 배경 사각형 그리기
                    cv2.rectangle(annotated_img, (10, 10), (20 + text_width, 20 + text_height + baseline), bg_color, -1)
                    
                    # FPS 텍스트 그리기
                    cv2.putText(annotated_img, fps_text, (15, 15 + text_height), font, font_scale, text_color, font_thickness)
                    
                    # 실시간 표시
                    if show_video:
                        cv2.imshow('YOLO Inference', annotated_img)
                        # 'q' 키를 누르면 종료
                        if cv2.waitKey(33) & 0xFF == ord('q'):
                            print("[INFO] User interrupted visualization")
                            break
                    
                    # 비디오 파일에 저장
                    if save_video and video_writer is not None:
                        video_writer.write(annotated_img)
                
                if is_video:
                    # 비디오는 매 프레임마다 로그
                    if num_images % 30 == 0:
                        print(f"[INFO] Processed {num_images} frames...")
                else:
                    # 이미지는 100장마다 로그
                    if num_images % 100 == 0:
                        print(f"[INFO] Processed {num_images} images...")
            
            # 리소스 정리
            if show_video:
                cv2.destroyAllWindows()
            if save_video and video_writer is not None:
                video_writer.release()
                print(f"[INFO] Video saved to {video_path}")
            
            total_time = total_preprocess + total_inference + total_postprocess
            avg_preprocess = total_preprocess / num_images if num_images > 0 else 0
            avg_inference = total_inference / num_images if num_images > 0 else 0
            avg_postprocess = total_postprocess / num_images if num_images > 0 else 0
            avg_total = total_time / num_images if num_images > 0 else 0
            fps = 1000 / avg_total if avg_total > 0 else 0

            print(f"\n[INFO] Inference Statistics:")
            print(f"  Total {'frames' if is_video else 'images'}: {num_images}")
            print(f"  Average preprocess: {avg_preprocess:.2f} ms")
            print(f"  Average inference: {avg_inference:.2f} ms")
            print(f"  Average postprocess: {avg_postprocess:.2f} ms")
            print(f"  Average total per {'frame' if is_video else 'image'}: {avg_total:.2f} ms")
            print(f"  FPS: {fps:.2f}")

            print(f"[INFO] Inference completed. Results saved to {save_dir}")
            
            # 통계 저장
            # 순서: 성능 관련 -> config 파라미터 -> 메타 정보
            stats['total_items'] = num_images
            stats['avg_preprocess'] = avg_preprocess
            stats['avg_inference'] = avg_inference
            stats['avg_postprocess'] = avg_postprocess
            stats['fps'] = fps
            stats.update(inference_params)
            stats.update(meta_info)
        
        # CSV 파일에 통계 저장
        if stats is not None:
            csv_path = args.csv_path
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            # 파일이 존재하는지 확인
            file_exists = os.path.isfile(csv_path)
            
            # CSV 파일에 쓰기
            with open(csv_path, 'a', newline='') as csvfile:
                # 헤더 생성 (config의 모든 키 + 통계 정보)
                fieldnames = list(stats.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # 파일이 없으면 헤더 작성
                if not file_exists:
                    writer.writeheader()
                
                # 데이터 행 작성
                writer.writerow(stats)
            
            print(f"[INFO] Statistics saved to {csv_path}")
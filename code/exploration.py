import os
import json
import itertools
import subprocess
from datetime import datetime

def generate_config_combinations():
    """
    다양한 predict 파라미터 조합을 생성하는 함수
    """
    # 테스트할 파라미터 값들
    param_grid = {
        'device': [0],  # GPU 디바이스
        'half': [True, False],  # FP16 사용 여부
        'imgsz': [640, 1280],  # 입력 이미지 크기
        'conf': [0.25, 0.5, 0.75],  # confidence threshold
        'iou': [0.7, 0.45, 0.9],  # IoU threshold for NMS
        'max_det': [50, 100, 300, 500],  # 최대 detection 수
        'batch': [1],  # 배치 크기
        'augment': [False, True],  # test-time augmentation
        'agnostic_nms': [False, True],  # class-agnostic NMS
    }
    
    # 고정 파라미터 (변경하지 않을 값들)
    fixed_params = {
        'stream': False,
        'verbose': False,
        'save_txt': True,
        'save_conf': False,
        'save_crop': False,
        'exist_ok': True,
        'show_video': False,
        'save_video': False,
        'video_fps': 30,
        'classes': None,
    }
    
    # 모든 조합 생성
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    configs = []
    for i, combo in enumerate(combinations):
        config = fixed_params.copy()
        for key, value in zip(keys, combo):
            config[key] = value
        
        config['config_id'] = i
        configs.append(config)
    
    return configs


def generate_selective_configs():
    """
    선택적으로 중요한 파라미터 조합만 생성 (실험 시간 단축)
    """
    configs = []
    config_id = 0
    
    # 1. 기본 설정 (baseline)
    baseline = {
        'device': 0,
        'half': True,
        'imgsz': 640,
        'conf': 0.25,
        'iou': 0.7,
        'max_det': 300,
        'batch': 1,
        'stream': False,
        'verbose': False,
        'augment': False,
        'agnostic_nms': False,
        'save_txt': True,
        'save_conf': False,
        'save_crop': False,
        'exist_ok': True,
        'show_video': False,
        'save_video': False,
        'video_fps': 30,
        'classes': None,
        'config_id': config_id,
    }
    configs.append(baseline.copy())
    config_id += 1
    
    # 2. Half precision 비교
    for half in [True, False]:
        config = baseline.copy()
        config['half'] = half
        config['config_id'] = config_id
        configs.append(config)
        config_id += 1
    
    # 3. Image size 비교
    # for imgsz in [320, 640, 1280]:
    #     config = baseline.copy()
    #     config['imgsz'] = imgsz
    #     config['config_id'] = config_id
    #     configs.append(config)
    #     config_id += 1
    
    # 4. Batch size 비교
    for batch in [1, 8, 16, 32]:
        config = baseline.copy()
        config['batch'] = batch
        config['config_id'] = config_id
        configs.append(config)
        config_id += 1
    
    # 5. Confidence threshold 비교
    for conf in [0.1, 0.25, 0.5, 0.75]:
        config = baseline.copy()
        config['conf'] = conf
        config['config_id'] = config_id
        configs.append(config)
        config_id += 1
    
    # 6. IoU threshold 비교
    for iou in [0.45, 0.7, 0.9]:
        config = baseline.copy()
        config['iou'] = iou
        config['config_id'] = config_id
        configs.append(config)
        config_id += 1
    
    # 7. Augmentation 비교
    for augment in [False, True]:
        config = baseline.copy()
        config['augment'] = augment
        config['config_id'] = config_id
        configs.append(config)
        config_id += 1
    
    # 8. Agnostic NMS 비교
    for agnostic_nms in [False, True]:
        config = baseline.copy()
        config['agnostic_nms'] = agnostic_nms
        config['config_id'] = config_id
        configs.append(config)
        config_id += 1
    
    # 9. 최적화된 조합 (속도 최적화)
    # speed_optimized = baseline.copy()
    # speed_optimized.update({
    #     'half': True,
    #     'imgsz': 320,
    #     'batch': 32,
    #     'conf': 0.5,
    #     'config_id': config_id,
    # })
    # configs.append(speed_optimized)
    # config_id += 1
    
    # 10. 최적화된 조합 (정확도 최적화)
    # accuracy_optimized = baseline.copy()
    # accuracy_optimized.update({
    #     'half': False,
    #     'imgsz': 640,
    #     'batch': 1,
    #     'conf': 0.1,
    #     'augment': True,
    #     'config_id': config_id,
    # })
    # configs.append(accuracy_optimized)
    
    return configs


def save_configs(configs, output_dir='configs'):
    """
    생성된 config를 JSON 파일로 저장
    """
    os.makedirs(output_dir, exist_ok=True)
    
    config_files = []
    for config in configs:
        config_id = config['config_id']
        filename = f"config_{config_id:04d}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        
        config_files.append(filepath)
        print(f"[INFO] Saved config to {filepath}")
    
    return config_files


def run_inference_batch(config_files, model_path, data_path, save_root_dir='exp', 
                       csv_path='exp/speed.csv', run_mode='iterative',
                       user='hwkang'):
    """
    모든 config 파일에 대해 inference.py를 실행
    """
    total = len(config_files)
    
    for idx, config_file in enumerate(config_files, 1):
        print(f"\n{'='*80}")
        print(f"[INFO] Running experiment {idx}/{total}")
        print(f"[INFO] Config file: {config_file}")
        print(f"{'='*80}\n")
        
        # inference.py 실행 명령어
        cmd = [
            'python', 'code/inference.py',
            '--mode', 'predict',
            '--model_path', model_path,
            '--data_path', data_path,
            '--config_path', config_file,
            '--save_root_dir', save_root_dir,
            '--run_mode', run_mode,
            '--user', user,
            '--suffix', f"config_{os.path.basename(config_file).split('.')[0]}",
        ]
        
        try:
            # subprocess로 실행
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            
            if result.stderr:
                print(f"[WARNING] Stderr output:\n{result.stderr}")
                
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to run inference for {config_file}")
            print(f"[ERROR] Error message: {e.stderr}")
            continue
        
        print(f"\n[INFO] Completed {idx}/{total} experiments")
    
    print(f"\n{'='*80}")
    print(f"[INFO] All experiments completed!")
    print(f"[INFO] Results saved to {csv_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Parameter Exploration Script")
    parser.add_argument('--mode', type=str, default='selective', 
                       choices=['full', 'selective'],
                       help='Config generation mode: full grid search or selective combinations')
    parser.add_argument('--model_path', type=str, 
                       default='/home/itec/mobility_project1/model/enric_yolo11l_20251124_150213/weights/best.engine',
                       help='Path to YOLO model')
    parser.add_argument('--data_path', type=str,
                       default='/home/itec/mobility_project1/data/test/image_2_mini',
                       help='Path to test dataset')
    parser.add_argument('--config_dir', type=str, default='env',
                       help='Directory to save config files')
    parser.add_argument('--csv_path', type=str, default='exp/speed.csv',
                       help='Path to save exploration results CSV')
    parser.add_argument('--save_root_dir', type=str, default='exp',
                       help='Root directory to save inference results')
    parser.add_argument('--run_mode', type=str, default='iterative',
                       choices=['single', 'iterative'],
                       help='Inference run mode')
    parser.add_argument('--user', type=str, default='hwkang',
                       help='User name for results')
    parser.add_argument('--generate_only', action='store_true',
                       help='Only generate config files without running inference')
    
    args = parser.parse_args()
    
    print("[INFO] Starting YOLO parameter exploration...")
    print(f"[INFO] Mode: {args.mode}")
    
    # Config 생성
    if args.mode == 'full':
        print("[INFO] Generating full grid search configurations...")
        configs = generate_config_combinations()
    else:
        print("[INFO] Generating selective configurations...")
        configs = generate_selective_configs()
    
    print(f"[INFO] Total configurations generated: {len(configs)}")
    
    # Config 파일 저장
    config_files = save_configs(configs, args.config_dir)
    
    if args.generate_only:
        print("[INFO] Config generation completed. Exiting without running inference.")
    else:
        # Inference 실행
        print("\n[INFO] Starting batch inference...")
        run_inference_batch(
            config_files=config_files,
            model_path=args.model_path,
            data_path=args.data_path,
            save_root_dir=args.save_root_dir,
            csv_path=args.csv_path,
            run_mode=args.run_mode,
            user=args.user,
        )
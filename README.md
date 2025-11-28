# Mobility Team Project 1

## Objective
- ADAS 기반 안전기능 구현
- KITTI 2D Object Detection Database 기반 객체 탐지 모델 개발

## Directory Architecture
```
mobility_project1/
 ├─ code/   : Codes related to the project
 ├─ docs/   : Documents related to the project
 ├─ env/    : Files for environmental setup
 ├─ data/   : Train/Validation Dataset (Symbolic link🔗)
 ├─ exp/    : Evaluation Result (Symbolic link🔗)
 ├─ log/    : Any logs occurred during execution
 └─ model/  : Store results of YOLO model training (Symbolic link🔗)
```

## Installation Guide
```
git clone https://github.com/pioneer0413/mobility_project1 \
cd mobility_project1/ \
bash init.sh \
source .venv/bin/activate \
pip install -r requirements.txt
```
```
# Now, you can see like below:
(.venv) user@Server:~/mobility_project1$ 
```
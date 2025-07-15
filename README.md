

# Brain Tumor Image Preprocessing

이 저장소는 뇌종양 CT 이미지 데이터셋을 다양한 이미지 전처리 기법을 적용하여 증강 및 전처리하는 파이썬 스크립트를 포함합니다.

## 주요 기능

- Z-score 정규화
- CLAHE (국소 대비 향상)
- Gaussian Blur (가우시안 블러)
- 이미지 샤프닝 (선명화)
- 밝기 조절
- 히스토그램 평활화
- 가우시안 노이즈 추가
- YOLO 형식 라벨 복사 및 관리
- | 기법명                       | 설명                               | 함수명                        |
| ------------------------- | -------------------------------- | -------------------------- |
| **Z-score 정규화**           | 이미지 픽셀 값을 평균 0, 표준편차 1로 스케일링     | `normalize_zscore()`       |
| **CLAHE (적응형 히스토그램 평활화)** | 국소 대비 향상, 의료 영상에서 유용             | `apply_clahe()`            |
| **가우시안 블러**               | 노이즈 제거 및 부드럽게 처리, 데이터 증강용        | `apply_gaussian_blur()`    |
| **샤프닝 필터**                | 경계 및 세부 구조 강조                    | `sharpen_image()`          |
| **밝기 조절**                 | 이미지 밝기 증가, 데이터 증강용               | `adjust_brightness()`      |
| **히스토그램 평활화**             | 이미지 전체 대비 향상 (밝기 분포 균일화)         | `histogram_equalization()` |
| **가우시안 노이즈 추가**           | Gaussian Noise 추가로 모델의 일반화 능력 향상 | `add_gaussian_noise()`     |


## 사용법

1. 원본 이미지와 라벨 폴더 경로를 `preprocess.py` 내부에서 설정합니다.
2. 터미널이나 커맨드라인에서 다음 명령어로 실행합니다:

```bash
python preprocess.py
전처리된 이미지와 라벨이 지정한 출력 폴더에 저장됩니다.

요구사항
Python 3.x

OpenCV (cv2)

NumPy

Pillow (PIL)

설치:

bash

pip install opencv-python numpy pillow




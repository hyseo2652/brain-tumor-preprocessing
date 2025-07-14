

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
복사
편집
pip install opencv-python numpy pillow


git add examples/
git commit -m "Add example images"
git push

import os

# 📁 복사된 이미지가 있는 폴더
examples_folder = "examples"

# 📷 이미지 파일만 필터링
image_files = [f for f in os.listdir(examples_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 🖼️ 마크다운 출력
print("\n## 예시 전처리 이미지\n")
for filename in image_files:
    print(f"![]({examples_folder}/{filename})")


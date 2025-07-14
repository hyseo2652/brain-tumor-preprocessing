# 전처리 전 데이터 수
    #트레이닝 세트: 893개의 이미지로 구성되며, 각 이미지에는 해당 주석이 첨부되어 있습니다.
    #테스트 세트: 223개의 이미지로 구성되며 각 이미지에 대한 주석이 함께 제공됩니다.

# 트레이닝 세트의 이미지만 사용

# 전처리 후 데이터 수
    
    # ===== 뇌종양 데이터 클래스 통계 =====
    #양성 이미지 수 (클래스 1): 3213
    #음성 이미지 수 (클래스 0): 2933
    #라벨 없는 이미지 수: 105
    #총 이미지 수: 6251
    
#클래스
    # 0: negative
    # 1: positive

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance

# ✅ Z-score 정규화
def normalize_zscore(img):
    img = img.astype(np.float32)
    mean = np.mean(img)
    std = np.std(img) + 1e-8
    return (img - mean) / std

# ✅ CLAHE (국소 대비 향상)
def apply_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# ✅ Gaussian Blur
def apply_gaussian_blur(img, ksize=(3, 3)):
    return cv2.GaussianBlur(img, ksize, 0)

# ✅ Sharpening
def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

# ✅ Brightness Adjustment
def adjust_brightness(img, factor=1.2):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_img)
    bright_img = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(bright_img), cv2.COLOR_RGB2BGR)

# ✅ Histogram Equalization
def histogram_equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# ✅ Add Gaussian Noise
def add_gaussian_noise(img, mean=0, std=10):
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

# ✅ YOLO 라벨 복사 함수
def copy_label_file(label_input_folder, base_name, suffix, label_output_folder):
    input_label_path = os.path.join(label_input_folder, base_name + ".txt")
    output_label_path = os.path.join(label_output_folder, f"{base_name}_{suffix}.txt")

    if os.path.exists(input_label_path):
        with open(input_label_path, "r") as infile:
            label_data = infile.read()
        os.makedirs(label_output_folder, exist_ok=True)
        with open(output_label_path, "w") as outfile:
            outfile.write(label_data)
    else:
        print(f"[경고] 라벨 없음: {input_label_path}")

# ✅ 이미지 전처리 및 저장
def process_and_save_all_images(image_input_folder, label_input_folder, image_output_folder, label_output_folder):
    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(label_output_folder, exist_ok=True)

    for filename in os.listdir(image_input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_input_folder, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"[경고] 이미지를 열 수 없음: {img_path}")
                continue

            base_name = os.path.splitext(filename)[0]
            print(f"[INFO] 처리 중: {filename}")

            try:
                # 전처리
                transformations = {
                    "normalized": normalize_zscore(img),
                    "clahe": apply_clahe(img),
                    "blurred": apply_gaussian_blur(img),
                    "sharpened": sharpen_image(img),
                    "bright": adjust_brightness(img),
                    "histogram_eq": histogram_equalization(img),
                    "noisy": add_gaussian_noise(img),
                }

                for suffix, processed_img in transformations.items():
                    output_img_path = os.path.join(image_output_folder, f"{base_name}_{suffix}.jpg")

                    # 저장 시 uint8 형 변환 필요 시 수행
                    if processed_img.dtype != np.uint8:
                        processed_img = cv2.normalize(processed_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                    cv2.imwrite(output_img_path, processed_img)

                    # 라벨 저장
                    copy_label_file(label_input_folder, base_name, suffix, label_output_folder)

            except Exception as e:
                print(f"[에러] {filename} 처리 중 문제 발생: {e}")

# ✅ 실행
if __name__ == "__main__":
    image_input_folder = "datasets/brain-tumor/train/images"     # 원본 이미지 폴더
    label_input_folder = "datasets/brain-tumor/train/labels"     # 원본 라벨 폴더
    image_output_folder = "processed_train_images"                     # 전처리된 이미지 저장
    label_output_folder = "processed_train_labels"                     # 전처리된 라벨 저장

    process_and_save_all_images(image_input_folder, label_input_folder, image_output_folder, label_output_folder)

print('finish')

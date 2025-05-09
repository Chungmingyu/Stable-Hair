# hair_refinement.py
import os
import numpy as np
import torch
import cv2
from PIL import Image
import sys

# SAM-HQ 경로 설정
sam_hq_dir = "sam-hq"
sys.path.append(sam_hq_dir)
from segment_anything import sam_model_registry, SamPredictor

def refine_hair_mask(image_path, hair_mask_path, output_dir):
    """
    Face-Parsing에서 생성된 머리카락 마스크를 SAM-HQ로 정교화
    """
    # SAM-HQ 모델 로드
    sam_checkpoint = "sam-hq/pretrained_checkpoint/sam_hq_vit_l.pth"
    model_type = "vit_l"
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Face-Parsing에서 생성된 머리카락 마스크 로드
    hair_mask = cv2.imread(hair_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # SAM-HQ 설정
    predictor.set_image(image)
    
    # 머리카락 마스크에서 포인트 추출 (머리카락 영역 내부 점들)
    y_indices, x_indices = np.where(hair_mask > 128)
    
    if len(y_indices) == 0:
        print("머리카락 마스크에서 포인트를 찾을 수 없습니다.")
        return hair_mask_path
    
    # 포인트 샘플링 (최대 10개)
    num_points = min(10, len(y_indices))
    idx = np.random.choice(len(y_indices), num_points, replace=False)
    
    input_points = np.array([[x_indices[i], y_indices[i]] for i in idx])
    input_labels = np.ones(num_points)  # 모든 포인트를 foreground로 표시
    
    # SAM-HQ 추론
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )
    
    # 가장 높은 점수의 마스크 선택
    best_mask_idx = np.argmax(scores)
    refined_mask = masks[best_mask_idx]
    
    # 결과 저장
    filename = os.path.basename(image_path).split('.')[0]
    os.makedirs(output_dir, exist_ok=True)
    
    # 바이너리 마스크로 변환
    refined_mask = refined_mask.astype(np.uint8) * 255
    
    # 원본 마스크와 정교화된 마스크 결합 (교집합)
    combined_mask = cv2.bitwise_and(hair_mask, refined_mask)
    
    # 결과 저장
    refined_hair_mask_path = os.path.join(output_dir, f"{filename}_refined_hair_mask.png")
    cv2.imwrite(refined_hair_mask_path, combined_mask)
    
    # 시각화를 위한 결과
    mask_vis = np.zeros_like(image)
    mask_vis[combined_mask > 0] = [0, 255, 0]  # 녹색으로 머리카락 표시
    vis_result = cv2.addWeighted(image, 0.7, mask_vis, 0.3, 0)
    
    vis_path = os.path.join(output_dir, f"{filename}_refined_hair_vis.png")
    cv2.imwrite(vis_path, cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR))
    
    return refined_hair_mask_path, vis_path


def remove_outside_hair_face(image_path, hair_mask_path, output_dir):
    """
    머리카락과 얼굴은 보존하고 양옆과 위의 바깥 영역만 제거
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # 머리카락 마스크 로드
    hair_mask = cv2.imread(hair_mask_path, cv2.IMREAD_GRAYSCALE)

    if hair_mask.shape != (h, w):
        hair_mask = cv2.resize(hair_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 중요: 머리카락 마스크의 내부 채우기
    # 윤곽선만 있는 경우를 대비하여 내부를 채움
    contours, _ = cv2.findContours(hair_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled_hair_mask = np.zeros_like(hair_mask)
    for contour in contours:
        cv2.drawContours(filled_hair_mask, [contour], 0, 255, -1)  # -1은 내부 채우기

    # 내부 홀 채우기 (morphological close 연산)
    kernel = np.ones((13, 13), np.uint8)  # 작은 커널로 미세한 홀만 채움
    filled_hair_mask = cv2.morphologyEx(filled_hair_mask, cv2.MORPH_CLOSE, kernel)
    
    # 채워진 머리카락 영역이 너무 적으면 원본 사용
    if np.sum(filled_hair_mask) < np.sum(hair_mask):
        filled_hair_mask = hair_mask
    
    # 머리카락 영역 확장 (경계 포함)
    kernel = np.ones((3, 3), np.uint8)
    hair_mask_dilated = cv2.dilate(filled_hair_mask, kernel, iterations=1)
    
    # 얼굴 영역 마스크 생성 (머리카락 아래부터)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 머리카락 마스크의 각 열(column)에서 가장 낮은 점 찾기
    for col in range(w):
        hair_pixels = np.where(hair_mask_dilated[:, col] > 0)[0]
        if len(hair_pixels) > 0:
            bottom_y = np.max(hair_pixels)
            # 머리카락 아래쪽부터 이미지 하단까지를 얼굴 영역으로
            face_mask[bottom_y:, col] = 255
    
    # 보존할 영역 마스크 (머리카락 + 얼굴)
    keep_mask = cv2.bitwise_or(hair_mask_dilated, face_mask)
    
    # 결과 이미지 생성
    result = image.copy()
    # 보존 마스크의 반전 영역(바깥쪽)을 검은색으로
    result[keep_mask == 0] = 0
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path).split('.')[0]
    result_path = os.path.join(output_dir, f"{filename}_hair_face_only.png")
    cv2.imwrite(result_path, result)
    
    # 디버깅용 마스크 저장
    mask_path = os.path.join(output_dir, f"{filename}_keep_mask.png")
    cv2.imwrite(mask_path, keep_mask)
    
    return result_path, mask_path

# 사용 예시:
# remove_outside_hair_face('output.png', 'output_refined_hair_mask.png', 'output_dir')
# refine_hair_mask('output.png', 'output_hair_mask.png', 'output_dir')
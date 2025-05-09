import os
import cv2
import numpy as np
from modules.simple_processor import FaceProcessor
from modules.background_removal import remove_background
from modules.face_parsing import get_face_parsing
from modules.hair_refinement import refine_hair_mask, remove_outside_hair_face


def process_image_pipeline(
    image_path,
    output_dir,
    target_size=1024,
    align_face=True,
    remove_bg=True,
    do_face_parsing=True,
    refine_hair=True,
    keep_hair_face_only=True,
    white_bg=True
):
    """
    전체 전처리 파이프라인 실행 함수
    Args:
        image_path (str): 입력 이미지 경로
        output_dir (str): 결과 저장 디렉토리
        target_size (int): 최종 리사이즈 크기 (기본 1024)
        align_face (bool): 얼굴 정렬 수행 여부
        remove_bg (bool): 배경 제거 수행 여부
        do_face_parsing (bool): face parsing 수행 여부
        refine_hair (bool): 머리카락 마스크 정교화 수행 여부
        keep_hair_face_only (bool): 얼굴+머리만 남기기 수행 여부
        white_bg (bool): 하얀 배경 이미지 생성 여부
    Returns:
        dict: 각 단계별 결과 파일 경로
    """
    os.makedirs(output_dir, exist_ok=True)
    result = {}
    
    # 1. 얼굴 정렬 및 리사이즈
    fp = FaceProcessor()
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지 로드 실패: {image_path}")
    if align_face:
        img = fp.align_face(img)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    aligned_path = os.path.join(output_dir, "aligned_resized.jpg")
    cv2.imwrite(aligned_path, img)
    result["aligned_resized"] = aligned_path

    # 2. 배경 제거
    if remove_bg:
        bg_removed_path = os.path.join(output_dir, "bg_removed.png")
        bg_removed_path, white_bg_path, mask_path = remove_background(aligned_path, bg_removed_path)
        result["bg_removed"] = bg_removed_path
        result["white_bg"] = white_bg_path
        result["bg_mask"] = mask_path
        img_for_parsing = white_bg_path if white_bg else bg_removed_path
    else:
        img_for_parsing = aligned_path

    # 3. Face Parsing (머리카락/얼굴/특징점 마스크)
    if do_face_parsing:
        parsing_result = get_face_parsing(img_for_parsing, output_dir)
        result.update(parsing_result)
        hair_mask_path = parsing_result["hair_mask"]
    else:
        hair_mask_path = None

    # 4. 머리카락 마스크 정교화
    if refine_hair and hair_mask_path:
        refined_hair_mask_path, vis_path = refine_hair_mask(img_for_parsing, hair_mask_path, output_dir)
        result["refined_hair_mask"] = refined_hair_mask_path
        result["refined_hair_vis"] = vis_path
    else:
        refined_hair_mask_path = hair_mask_path

    # 5. 얼굴+머리만 남기기 (팔 등 제거)
    if keep_hair_face_only and refined_hair_mask_path:
        hair_face_only_path, keep_mask_path = remove_outside_hair_face(img_for_parsing, refined_hair_mask_path, output_dir)
        result["hair_face_only"] = hair_face_only_path
        result["keep_mask"] = keep_mask_path

    return result 

def combine_parsing_masks(parsing_map_path, keep_classes, output_path):
    """
    parsing_map.png에서 원하는 클래스만 남기는 마스크 생성
    Args:
        parsing_map_path (str): face parsing에서 생성된 parsing_map.png 경로
        keep_classes (list): 남기고 싶은 클래스 인덱스 리스트 (예: [1, 2, 3, 7, 8, 16])
        output_path (str): 저장 경로
    Returns:
        str: 생성된 마스크 경로
    """
    parsing_map = cv2.imread(parsing_map_path, cv2.IMREAD_GRAYSCALE)
    mask = np.zeros_like(parsing_map, dtype=np.uint8)
    for cls in keep_classes:
        mask[parsing_map == cls] = 255
    cv2.imwrite(output_path, mask)
    return output_path

def black_to_white(image_path, output_path):
    """
    이미지에서 완전 검은색(0,0,0) 픽셀을 하얀색(255,255,255)으로 변환
    Args:
        image_path (str): 입력 이미지 경로
        output_path (str): 결과 저장 경로
    Returns:
        str: 저장된 이미지 경로
    """
    img = cv2.imread(image_path)
    mask = np.all(img == [0,0,0], axis=2)
    img[mask] = [255,255,255]
    cv2.imwrite(output_path, img)
    return output_path

def keep_non_hair_from_parsing(parsing_map_path, inpainted_img_path, white_bg_img_path, output_path):
    """
    parsing_map에서 머리카락(17)만 제외, 나머지 부분만 살려서 인페인팅 결과와 white bg 이미지를 합성
    Args:
        parsing_map_path (str): 파싱 인덱스맵 (png, 0~18)
        inpainted_img_path (str): 대머리 인페인팅 결과
        white_bg_img_path (str): 배경 제거된 하얀 배경 이미지
        output_path (str): 저장 경로
    Returns:
        str: 저장된 이미지 경로
    """
    parsing_map = cv2.imread(parsing_map_path, cv2.IMREAD_GRAYSCALE)
    inpainted = cv2.imread(inpainted_img_path)
    white_bg = cv2.imread(white_bg_img_path)
    # parsing_map을 white_bg와 같은 크기로 리사이즈
    parsing_map_resized = cv2.resize(parsing_map, (white_bg.shape[1], white_bg.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = np.ones_like(parsing_map_resized, dtype=np.uint8) * 255
    mask[parsing_map_resized == 17] = 0
    mask3 = cv2.merge([mask, mask, mask])
    result = np.where(mask3 == 255, white_bg, inpainted)
    cv2.imwrite(output_path, result)
    return output_path

def keep_parsing_parts_as_mask(parsing_map_path, inpainted_img_path, white_bg_img_path, output_path, keep_classes=None):
    """
    parsing_map에서 지정한 클래스(예: 얼굴, 옷, 귀, 눈썹 등)만 남기고 나머지는 하얀색으로 만듦
    Args:
        parsing_map_path (str): face-parsing에서 생성된 parsing_map.png 경로
        inpainted_img_path (str): 대머리 인페인팅 결과 이미지 경로
        white_bg_img_path (str): 배경 제거된 하얀 배경 이미지 경로
        output_path (str): 저장 경로
        keep_classes (list): 남길 클래스 인덱스 리스트 (기본값: [1,2,3,7,8,16])
    Returns:
        str: 저장된 이미지 경로
    """
    if keep_classes is None:
        keep_classes = [1,2,3,7,8,16]  # 얼굴, 눈썹, 귀, 옷 등
    parsing_map = cv2.imread(parsing_map_path, cv2.IMREAD_GRAYSCALE)
    inpainted = cv2.imread(inpainted_img_path)
    white_bg = cv2.imread(white_bg_img_path)
    # parsing_map을 white_bg와 같은 크기로 리사이즈
    parsing_map_resized = cv2.resize(parsing_map, (white_bg.shape[1], white_bg.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = np.zeros_like(parsing_map_resized, dtype=np.uint8)
    for cls in keep_classes:
        mask[parsing_map_resized == cls] = 255
    mask3 = cv2.merge([mask, mask, mask])
    # 지정된 부분은 inpainted, 나머지는 하얀색
    result = np.where(mask3 == 255, inpainted, 255)
    cv2.imwrite(output_path, result)
    return output_path 
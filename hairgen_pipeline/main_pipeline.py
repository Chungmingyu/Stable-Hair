import argparse
import os
from modules.preprocessing import process_image_pipeline, keep_parsing_parts_as_mask
from modules.hair_generation import generate_hair_pipeline
from modules.face_parsing import make_keep_face_body_mask_from_existing

def main():
    parser = argparse.ArgumentParser(description="자연스러운 헤어스타일 변환 파이프라인")
    parser.add_argument('--input', required=True, help='입력 이미지 경로')
    parser.add_argument('--output_dir', default='output_dir', help='결과 저장 디렉토리')
    parser.add_argument('--target_size', type=int, default=1024, help='최종 리사이즈 크기')
    parser.add_argument('--device', default=None, help='cuda/cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 입력 이미지 전처리 (최소 결과만 저장)
    print("[1/5] 입력 이미지 전처리 중...")
    preprocess_result = process_image_pipeline(
        args.input,
        args.output_dir,
        target_size=args.target_size,
        align_face=True,
        remove_bg=True,
        do_face_parsing=True,
        refine_hair=True,
        keep_hair_face_only=False,  # 팔/배경 제거는 마지막에
        white_bg=True
    )
    white_bg_img = preprocess_result['white_bg']
    parsing_map = preprocess_result['parsing_map']
    refined_hair_mask = preprocess_result.get('refined_hair_mask') or preprocess_result['hair_mask']
    face_mask = preprocess_result['face_mask']

    # 2. 머리카락 마스크로 대머리 인페인팅
    print("[2/5] 대머리 인페인팅 중...")
    bald_img_path, _ = generate_hair_pipeline(
        white_bg_img,
        refined_hair_mask,
        face_mask,
        None,
        args.output_dir,
        mode='bald',
        device=args.device,
        parsing_map_path=parsing_map,
        white_bg_image_path=white_bg_img
    )

    # 3. parsing_map에서 얼굴/옷/귀/눈썹 등만 남기고 나머지는 하얀색으로 합성
    print("[3/5] parsing 결과 기반 최종 합성 중...")
    final_img_path = os.path.join(args.output_dir, 'final_bald_with_body.png')
    keep_parsing_parts_as_mask(
        parsing_map,
        bald_img_path,
        white_bg_img,
        final_img_path,
        keep_classes=[1,2,3,4,5,7,8,10,11,12,13,14,15,16]  # 얼굴, 눈썹, 눈, 귀, 코, 입, 목, 목걸이, 옷 등
    )

    print(f"[완료] 최종 결과: {final_img_path}")

    # 5. 얼굴과 몸만 남긴 마스크에서 위쪽만 검정(0, 생성)으로 바꾸기
    print("[5/5] 얼굴과 몸만 남긴 마스크에서 위쪽만 검정(0, 생성)으로 바꾸기 중...")
    protect_mask_path = os.path.join(args.output_dir, 'protect_mask.png')
    make_keep_face_body_mask_from_existing(
        protect_mask_path,
        "output_keep_face_body_mask_eyebrow.png",
        ratio=0.38  # 필요에 따라 조정
    )

if __name__ == "__main__":
    main() 
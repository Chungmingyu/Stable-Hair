import os
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import argparse
from modules.preprocessing import combine_parsing_masks

def create_bald_image(image_path, hair_mask_path, face_mask_path, output_dir, 
                      model_id="stabilityai/stable-diffusion-2-inpainting", 
                      device=None,
                      parsing_map_path=None,
                      white_bg_image_path=None):
    """
    인페인팅을 사용하여 자연스러운 대머리 이미지 생성 (파싱 기반 보호 마스크 활용)
    Args:
        image_path (str): 입력 이미지 경로 (하얀 배경 합성본 권장)
        hair_mask_path (str): 머리카락 마스크 경로 (흰색=머리카락)
        face_mask_path (str): 얼굴 마스크 경로 (흰색=얼굴)
        output_dir (str): 결과물 저장 디렉토리
        model_id (str): 인페인팅 모델 ID
        device (str): 장치 (None=자동 감지)
        parsing_map_path (str): face parsing에서 생성된 parsing_map.png 경로
        white_bg_image_path (str): 하얀 배경 합성 이미지 경로 (있으면 이걸 사용)
    Returns:
        tuple: (대머리 이미지 경로, 비교 이미지 경로)
    """
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"인페인팅 모델 로드 중... ({model_id})")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)

    # 원본 이미지 로드 (하얀 배경 합성본이 있으면 그걸 사용)
    if white_bg_image_path is not None:
        original_image = Image.open(white_bg_image_path).convert("RGB")
    else:
        original_image = Image.open(image_path).convert("RGB")
    org_width, org_height = original_image.size

    # 파싱 맵에서 보호 마스크 생성 (얼굴, 눈썹, 귀, 옷 등)
    if parsing_map_path is not None:
        keep_classes = [1,2,3,4,5,7,8,10,11,12,13,14,15,16] 
        protect_mask_path = os.path.join(output_dir, "protect_mask.png")
        combine_parsing_masks(parsing_map_path, keep_classes, protect_mask_path)
        # 얼굴+눈썹+귀+옷 마스크 로드
        protect_mask = Image.open(protect_mask_path).convert("L")
        protect_mask = protect_mask.resize((org_width, org_height), Image.NEAREST)
    else:
        protect_mask = None

    # 머리카락 마스크 로드 (머리카락만 255, 나머지 0)
    hair_mask = Image.open(hair_mask_path).convert("L")
    if hair_mask.size != original_image.size:
        hair_mask = hair_mask.resize((org_width, org_height), Image.NEAREST)

    # 인페인팅 마스크: 머리카락만 255, 나머지는 0
    inpaint_mask = hair_mask

    # SD 모델에 맞게 리사이즈 및 패딩
    target_size = 512
    ratio = min(target_size / org_width, target_size / org_height)
    new_width = int(org_width * ratio)
    new_height = int(org_height * ratio)
    resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
    resized_inpaint_mask = inpaint_mask.resize((new_width, new_height), Image.NEAREST)
    if protect_mask is not None:
        resized_protect_mask = protect_mask.resize((new_width, new_height), Image.NEAREST)
    else:
        resized_protect_mask = None
    square_image = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    square_inpaint_mask = Image.new("L", (target_size, target_size), 0)
    square_protect_mask = Image.new("L", (target_size, target_size), 0) if resized_protect_mask is not None else None
    offset_x = (target_size - new_width) // 2
    offset_y = (target_size - new_height) // 2
    square_image.paste(resized_image, (offset_x, offset_y))
    square_inpaint_mask.paste(resized_inpaint_mask, (offset_x, offset_y))
    if square_protect_mask is not None:
        square_protect_mask.paste(resized_protect_mask, (offset_x, offset_y))

    # 인페인팅 프롬프트
    prompt = "a realistic bald head, smooth scalp with natural skin texture, photorealistic, high quality, detailed skin pores, clean shaved head"
    negative_prompt = "hair, fur, wig, hat, cap, unrealistic, cartoon, low quality, blurry, deformed"

    # 인페인팅 수행
    print("인페인팅으로 대머리 생성 중...")
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=square_image,
            mask_image=square_inpaint_mask,
            num_inference_steps=50,
            guidance_scale=7.5,
            width=target_size,
            height=target_size,
        ).images[0]

    # 패딩 제거 및 원본 크기로 복원
    cropped_output = output.crop((offset_x, offset_y, offset_x + new_width, offset_y + new_height))
    restored_output = cropped_output.resize((org_width, org_height), Image.LANCZOS)

    # 보호 마스크(얼굴/눈썹/귀/옷 등)는 원본 이미지로 복원
    if protect_mask is not None:
        output_np = np.array(restored_output)
        original_np = np.array(original_image)
        protect_mask_np = np.array(protect_mask)
        protect_mask_bin = protect_mask_np > 128
        protect_mask_bin = np.expand_dims(protect_mask_bin, axis=2).astype(np.float32)
        protect_mask_bin = np.repeat(protect_mask_bin, 3, axis=2)
        blended_output = output_np * (1 - protect_mask_bin) + original_np * protect_mask_bin
        final_output = blended_output.astype(np.uint8)
    else:
        final_output = np.array(restored_output)

    # 결과 이미지 저장
    filename = os.path.basename(image_path).split('.')[0]
    bald_image_path = os.path.join(output_dir, f"{filename}_bald_inpainted.png")
    Image.fromarray(final_output).save(bald_image_path)
    comparison = np.hstack((np.array(original_image), final_output))
    comparison_path = os.path.join(output_dir, f"{filename}_bald_comparison.jpg")
    Image.fromarray(comparison).save(comparison_path)
    print(f"대머리 이미지 생성 완료: {bald_image_path}")
    print(f"비교 이미지: {comparison_path}")
    return bald_image_path, comparison_path

def main():
    parser = argparse.ArgumentParser(description="인페인팅을 사용한 대머리 이미지 생성")
    parser.add_argument("--input", required=True, help="입력 이미지 경로")
    parser.add_argument("--hair_mask", required=True, help="머리카락 마스크 경로")
    parser.add_argument("--face_mask", required=True, help="얼굴 마스크 경로")
    parser.add_argument("--output", default="output", help="출력 디렉토리")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-2-inpainting", help="인페인팅 모델 ID")
    parser.add_argument("--device", help="장치 (cuda/cpu)")
    parser.add_argument("--parsing_map", help="face parsing에서 생성된 parsing_map.png 경로")
    parser.add_argument("--white_bg_image", help="하얀 배경 합성 이미지 경로")
    
    args = parser.parse_args()
    
    create_bald_image(
        args.input, 
        args.hair_mask, 
        args.face_mask, 
        args.output,
        args.model,
        args.device,
        args.parsing_map,
        args.white_bg_image
    )

if __name__ == "__main__":
    main()
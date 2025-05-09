import os
import json
import torch
import argparse
from PIL import Image
import numpy as np
import cv2
from diffusers import StableDiffusionInpaintPipeline

def generate_hairstyle(prep_dir, output_dir):
    """준비된 데이터로 헤어스타일 생성"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 설정 로드
    with open(os.path.join(prep_dir, "generation_config.json"), "r") as f:
        config = json.load(f)
    
    # 이미지 및 마스크 로드
    image = Image.open(config["canvas_path"])
    mask = Image.open(config["hair_mask_path"])
    face_mask = Image.open(config["face_mask_path"]) if "face_mask_path" in config else None
    
    # 프롬프트 사용 (필요시 수정)
    prompt = config["prompt"]
    print(f"사용 프롬프트: {prompt}")
    
    # 네거티브 프롬프트
    negative_prompt = "deformed, distorted, disfigured, bad anatomy, unrealistic, low quality, blurry, bad hair"
    
    # 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)
    
    # 생성 설정
    num_samples = 3  # 생성할 이미지 수
    num_inference_steps = 50
    guidance_scale = 7.5
    
    # 여러 이미지 생성
    for i in range(num_samples):
        # 시드 설정 (재현성)
        seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # 인페인팅 실행
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
        
        # 얼굴 복원 (필요한 경우)
        if face_mask is not None:
            result = restore_face(result, image, face_mask)
        
        # 원본 크기로 복원
        if "size_info" in config:
            result = restore_original_size(result, config["size_info"])
        
        # 결과 저장
        output_path = os.path.join(output_dir, f"result_{i+1}.png")
        result.save(output_path)
        
        # 메타데이터 저장
        metadata = {
            "prompt": prompt,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale
        }
        
        with open(os.path.join(output_dir, f"result_{i+1}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"이미지 {i+1} 생성 완료: {output_path}")

def restore_face(result_image, original_image, face_mask):
    """얼굴 영역 복원"""
    # PIL → numpy
    result_np = np.array(result_image)
    original_np = np.array(original_image)
    face_mask_np = np.array(face_mask)
    
    # 얼굴 마스크 이진화
    face_mask_bin = face_mask_np > 128
    
    # 경계 계산
    kernel = np.ones((5, 5), np.uint8)
    face_expanded = cv2.dilate(face_mask_np, kernel, iterations=2)
    face_boundary = face_expanded - face_mask_np
    face_boundary = (face_boundary > 0)
    
    # 채널 확장
    face_mask_bin = np.stack([face_mask_bin] * 3, axis=2)
    face_boundary = np.stack([face_boundary] * 3, axis=2)
    
    # 얼굴 복원
    blended = np.where(face_mask_bin, original_np, result_np)
    
    # 경계 블렌딩
    try:
        blended_cv = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        blurred = cv2.GaussianBlur(blended_cv, (5, 5), 0)
        blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        
        final_result = np.where(face_boundary, blurred, blended)
    except:
        final_result = blended
    
    return Image.fromarray(final_result.astype(np.uint8))

def restore_original_size(image, size_info):
    """원본 크기로 복원"""
    # 패딩 제거
    if "offset" in size_info:
        offset_x, offset_y = size_info["offset"]
        resized_w, resized_h = size_info["resized_size"]
        
        cropped = image.crop((offset_x, offset_y, offset_x + resized_w, offset_y + resized_h))
    else:
        cropped = image
    
    # 원본 크기로 리사이즈
    if "original_size" in size_info:
        orig_w, orig_h = size_info["original_size"]
        return cropped.resize((orig_w, orig_h), Image.LANCZOS)
    
    return cropped

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="헤어스타일 이미지 생성")
    parser.add_argument("--prep_dir", default="hairstyle_prep", help="준비 디렉토리")
    parser.add_argument("--output", default="hairstyle_results", help="출력 디렉토리")
    
    args = parser.parse_args()
    generate_hairstyle(args.prep_dir, args.output)
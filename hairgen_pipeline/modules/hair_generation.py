import os
from modules.bald_generation import create_bald_image

# (실제 IP-Adapter XL inference는 추후 구현)
def generate_hairstyle_with_ipadapter(
    image_path,
    hair_mask_path,
    face_mask_path,
    reference_image_path,
    output_dir,
    model_id="ip-adapter-xl",
    device=None
):
    """
    IP-Adapter XL을 사용한 헤어스타일 생성 (비대머리)
    Args:
        image_path (str): 입력 이미지 경로
        hair_mask_path (str): 머리카락 마스크 경로
        face_mask_path (str): 얼굴 마스크 경로
        reference_image_path (str): 참고 이미지 경로
        output_dir (str): 결과 저장 디렉토리
        model_id (str): IP-Adapter XL 모델 ID
        device (str): 장치
    Returns:
        str: 생성된 이미지 경로
    """
    # TODO: IP-Adapter XL inference 구현
    # 현재는 placeholder
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hairstyle_generated.png")
    # ... IP-Adapter XL inference 코드 ...
    print("[알림] IP-Adapter XL 기반 헤어스타일 생성은 추후 구현 필요!")
    return output_path


def generate_hair_pipeline(
    image_path,
    hair_mask_path,
    face_mask_path,
    reference_image_path,
    output_dir,
    mode="bald",
    inpaint_model_id="stabilityai/stable-diffusion-2-inpainting",
    ipadapter_model_id="ip-adapter-xl",
    device=None,
    parsing_map_path=None,
    white_bg_image_path=None
):
    """
    대머리/비대머리 분기 파이프라인
    Args:
        image_path (str): 입력 이미지 경로
        hair_mask_path (str): 머리카락 마스크 경로
        face_mask_path (str): 얼굴 마스크 경로
        reference_image_path (str): 참고 이미지 경로 (비대머리용)
        output_dir (str): 결과 저장 디렉토리
        mode (str): 'bald' 또는 'hairstyle'
        inpaint_model_id (str): 인페인팅 모델 ID
        ipadapter_model_id (str): IP-Adapter XL 모델 ID
        device (str): 장치
        parsing_map_path (str): face parsing에서 생성된 parsing_map.png 경로
        white_bg_image_path (str): 하얀 배경 합성 이미지 경로
    Returns:
        str: 생성된 이미지 경로
    """
    if mode == "bald":
        return create_bald_image(
            image_path,
            hair_mask_path,
            face_mask_path,
            output_dir,
            model_id=inpaint_model_id,
            device=device,
            parsing_map_path=parsing_map_path,
            white_bg_image_path=white_bg_image_path
        )
    elif mode == "hairstyle":
        return generate_hairstyle_with_ipadapter(
            image_path,
            hair_mask_path,
            face_mask_path,
            reference_image_path,
            output_dir,
            model_id=ipadapter_model_id,
            device=device
        )
    else:
        raise ValueError(f"지원하지 않는 mode: {mode}") 
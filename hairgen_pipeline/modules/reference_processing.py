from modules.preprocessing import process_image_pipeline

def process_reference_image(
    reference_image_path,
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
    레퍼런스(참고) 이미지 전처리 파이프라인
    Args:
        reference_image_path (str): 레퍼런스 이미지 경로
        output_dir (str): 결과 저장 디렉토리
        ... (기타 옵션은 process_image_pipeline과 동일)
    Returns:
        dict: 각 단계별 결과 파일 경로
    """
    return process_image_pipeline(
        reference_image_path,
        output_dir,
        target_size=target_size,
        align_face=align_face,
        remove_bg=remove_bg,
        do_face_parsing=do_face_parsing,
        refine_hair=refine_hair,
        keep_hair_face_only=keep_hair_face_only,
        white_bg=white_bg
    ) 
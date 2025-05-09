from modules.eva_clip_embedder import process_reference_directory_with_eva_clip

result = process_reference_directory_with_eva_clip(
    reference_dir='reference_dir',  # 테스트용 이미지 폴더
    output_dir='reference_dir/processed',  # 결과 저장 폴더
    model_path='models/EVA02_CLIP_L_336_psz14_s6B.pt',
    device='cuda'  # 또는 'cpu'
)

print(result)
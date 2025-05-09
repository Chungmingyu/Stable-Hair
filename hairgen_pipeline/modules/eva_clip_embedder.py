import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T
from modules.face_parsing import get_face_parsing
import glob
import os

class EVAClipEmbedder:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()
        self.model.to(device)
        # EVA-CLIP 공식 전처리 파이프라인 (CLIP과 유사)
        self.preprocess = T.Compose([
            T.Resize((336, 336), interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def get_embedding(self, image, hair_mask=None):
        """
        image: PIL.Image
        hair_mask: PIL.Image or None (흑백, 머리 부분만 255)
        return: numpy array (임베딩)
        """
        if hair_mask is not None:
            # 머리 부분만 남기고 나머지는 흰색(255)으로 처리
            image = image.copy()
            arr = np.array(image)
            mask = np.array(hair_mask).astype(bool)
            white = np.ones_like(arr) * 255
            arr = np.where(mask[..., None], arr, white)
            image = Image.fromarray(arr)
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # EVA-CLIP은 일반적으로 model.encode_image 사용
            if hasattr(self.model, 'encode_image'):
                emb = self.model.encode_image(img_tensor)
            else:
                emb = self.model(img_tensor)
            emb = emb.cpu().numpy().squeeze()
        return emb 

def process_reference_directory_with_eva_clip(reference_dir, output_dir, model_path, device='cuda'):
    """
    reference_dir: 참고 이미지가 들어있는 폴더
    output_dir: 결과 저장 폴더
    model_path: EVA-CLIP 모델 경로
    device: 'cuda' or 'cpu'
    """
    os.makedirs(output_dir, exist_ok=True)
    embedder = EVAClipEmbedder(model_path, device=device)
    image_paths = glob.glob(os.path.join(reference_dir, '*'))
    results = {}
    for img_path in image_paths:
        if not img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            continue
        # 1. 얼굴 파싱 및 헤어 마스크 추출
        parsing_result = get_face_parsing(img_path, output_dir)
        hair_mask_path = parsing_result['hair_mask']
        hair_mask = Image.open(hair_mask_path).convert('L')
        image = Image.open(img_path).convert('RGB')
        # 2. EVA-CLIP 임베딩 추출
        emb = embedder.get_embedding(image, hair_mask=hair_mask)
        # 3. 임베딩 저장
        base = os.path.splitext(os.path.basename(img_path))[0]
        emb_path = os.path.join(output_dir, f'{base}_hair_emb.npy')
        np.save(emb_path, emb)
        results[img_path] = {
            'embedding_path': emb_path,
            'hair_mask_path': hair_mask_path,
            'parsing_result': parsing_result
        }
        print(f"{img_path} → 임베딩 저장: {emb_path}")
    return results 
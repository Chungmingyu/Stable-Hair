from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. 모델 로드
from transformers import AutoModelForImageSegmentation
birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
birefnet.eval()
birefnet.to('cuda' if torch.cuda.is_available() else 'cpu')

def remove_background(image_path, output_path):
    # 2. 이미지 전처리
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(image.size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. 추론
    with torch.no_grad():
        output = birefnet(input_tensor)
        pred = output[-1].sigmoid().cpu().squeeze()

    # 4. 마스크 후처리 및 시각화
    # 이진화 및 8비트 변환
    threshold = 0.5
    binary_mask = (pred > threshold).float() * 255
    mask = binary_mask.byte().numpy()
    mask_pil = Image.fromarray(mask).resize(image.size, resample=Image.NEAREST).convert('L')

    # 파일명 자동 생성
    white_bg_path = output_path.replace('.png', '_white_bg.jpg')
    mask_path = output_path.replace('.png', '_mask.png')

    # 1. 알파 채널 PNG
    rgba = image.convert("RGBA")
    rgba.putalpha(mask_pil)
    rgba.save(output_path)

    # 2. 흰 배경 JPG
    white_bg = Image.new("RGB", image.size, (255, 255, 255))
    alpha = np.array(mask_pil) / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    img_np = np.array(image)
    result = (img_np * alpha + 255 * (1 - alpha)).astype(np.uint8)
    Image.fromarray(result).save(white_bg_path)

    # 3. 마스크 PNG
    mask_pil.save(mask_path)

    print(f"저장 완료:\n- 투명 PNG: {output_path}\n- 흰배경 JPG: {white_bg_path}\n- 마스크 PNG: {mask_path}")
    return output_path, white_bg_path, mask_path

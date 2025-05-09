# face_parsing.py
import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys

# 경로 설정
face_parsing_dir = "face-parsing.PyTorch"
sys.path.append(face_parsing_dir)
from model import BiSeNet

def get_face_parsing(image_path, output_dir):
    # 모델 로드
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    
    # CUDA 사용 가능 여부 확인
    if torch.cuda.is_available():
        net.cuda()
        save_pth = os.path.join(face_parsing_dir, 'res/cp/79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
    else:
        save_pth = os.path.join(face_parsing_dir, 'res/cp/79999_iter.pth')
        net.load_state_dict(torch.load(save_pth, map_location='cpu'))
    
    net.eval()
    
    # 입력 이미지 로드 및 전처리
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = image.resize((512, 512), Image.BILINEAR)
    img = to_tensor(image)
    img = torch.unsqueeze(img, 0)
    
    if torch.cuda.is_available():
        img = img.cuda()
    
    # 추론
    with torch.no_grad():
        out = net(img)[0]
    
    parsing = out.squeeze(0).cpu().numpy().argmax(0)
    
    # 결과 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 각 클래스에 대한 색상 맵 (시각화용)
    colors = [
        [0, 0, 0],        # 0: 배경
        [204, 0, 0],      # 1: 피부
        [76, 153, 0],     # 2: 왼쪽 눈썹
        [76, 153, 0],     # 3: 오른쪽 눈썹
        [0, 255, 255],    # 4: 왼쪽 눈
        [0, 255, 255],    # 5: 오른쪽 눈
        [255, 204, 204],  # 6: 안경
        [204, 0, 255],    # 7: 왼쪽 귀
        [204, 0, 255],    # 8: 오른쪽 귀
        [102, 51, 0],     # 9: 귀걸이
        [255, 0, 0],      # 10: 코
        [102, 204, 0],    # 11: 입
        [255, 255, 0],    # 12: 윗입술
        [0, 0, 153],      # 13: 아랫입술
        [0, 0, 204],      # 14: 목
        [255, 153, 51],   # 15: 목걸이
        [0, 204, 204],    # 16: 옷
        [76, 76, 76],     # 17: 머리카락
        [153, 0, 0],      # 18: 모자
    ]
    
    # 시각화 이미지 생성
    vis_parsing_anno = parsing.copy()
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3), dtype=np.uint8)
    
    for pi in range(19):
        vis_parsing_anno_color[vis_parsing_anno == pi] = colors[pi]
    
    vis_parsing_anno_color = cv2.resize(vis_parsing_anno_color, (image.size[0], image.size[1]), interpolation=cv2.INTER_NEAREST)
    vis_im = np.array(image)
    vis_im = cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR)
    vis_result = cv2.addWeighted(vis_im, 0.4, vis_parsing_anno_color, 0.6, 0)
    
    # 파일명 추출
    filename = os.path.basename(image_path).split('.')[0]
    
    # 결과 저장
    parsing_result_path = os.path.join(output_dir, f"{filename}_parsing.png")
    cv2.imwrite(parsing_result_path, vis_result)
    
    # 파싱 맵 저장 (분석용)
    parsing_map_path = os.path.join(output_dir, f"{filename}_parsing_map.png")
    cv2.imwrite(parsing_map_path, vis_parsing_anno)
    
    # 머리카락 마스크 추출 (클래스 17)
    hair_mask = np.zeros((parsing.shape[0], parsing.shape[1]), dtype=np.uint8)
    hair_mask[parsing == 17] = 255
    
    # 얼굴 마스크 추출 (클래스 1: 피부)
    face_mask = np.zeros((parsing.shape[0], parsing.shape[1]), dtype=np.uint8)
    face_mask[parsing == 1] = 255
    
    # 눈, 코, 입 마스크 (클래스 4, 5, 10, 11, 12, 13)
    facial_features_mask = np.zeros((parsing.shape[0], parsing.shape[1]), dtype=np.uint8)
    for cls in [2, 3, 4, 5, 7, 8, 10, 11, 12, 13]:  # 눈썹, 눈, 코, 입, 귀 클래스
        facial_features_mask[parsing == cls] = 255
    
    # 원본 크기로 리사이징
    original_size = Image.open(image_path).size
    hair_mask_resized = cv2.resize(hair_mask, original_size, interpolation=cv2.INTER_NEAREST)
    face_mask_resized = cv2.resize(face_mask, original_size, interpolation=cv2.INTER_NEAREST)
    facial_features_mask_resized = cv2.resize(facial_features_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    # 마스크 저장
    hair_mask_path = os.path.join(output_dir, f"{filename}_hair_mask.png")
    face_mask_path = os.path.join(output_dir, f"{filename}_face_mask.png")
    facial_features_mask_path = os.path.join(output_dir, f"{filename}_facial_features_mask.png")
    
    cv2.imwrite(hair_mask_path, hair_mask_resized)
    cv2.imwrite(face_mask_path, face_mask_resized)
    cv2.imwrite(facial_features_mask_path, facial_features_mask_resized)
    
    return {
        'parsing_result': parsing_result_path,
        'parsing_map': parsing_map_path,
        'hair_mask': hair_mask_path,
        'face_mask': face_mask_path,
        'facial_features_mask': facial_features_mask_path
    }


def make_keep_face_body_mask_from_existing(mask_path, output_path, ratio=0.38):
    """
    기존 마스크에서 위쪽(이마/머리/배경 등)만 검정(0, 생성)으로 바꾸고, 아래쪽(눈썹 아래+몸)은 흰색(255, 보호)으로 유지
    ratio: 위에서부터 몇 %까지를 검정(0, 생성)으로 바꿀지 (기본 0.38)
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape
    y_cut = int(h * ratio)
    mask[:y_cut, :] = 0  # 위쪽을 검정(0, 생성)
    cv2.imwrite(output_path, mask)
    print(f"눈썹 위쪽을 검정(생성)으로 바꾼 마스크 저장: {output_path}")
    return mask

# 사용 예시:
# get_face_parsing('output.png', 'output_dir')
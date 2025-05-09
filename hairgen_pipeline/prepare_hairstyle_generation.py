import os
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import argparse
import glob
from typing import List, Tuple, Dict, Optional, Union

class HairstyleGenerator:
    """헤어스타일 생성을 위한 클래스"""
    
    def __init__(self, device=None, 
                 inpaint_model_id="stabilityai/stable-diffusion-2-inpainting", 
                 img2img_model_id="stabilityai/stable-diffusion-2-1"):
        """
        헤어스타일 생성기 초기화
        
        Args:
            device: 계산 장치 (None=자동 감지)
            inpaint_model_id: 인페인팅 모델 ID
            img2img_model_id: 이미지-이미지 모델 ID
        """
        # 장치 설정
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.inpaint_model_id = inpaint_model_id
        self.img2img_model_id = img2img_model_id
        
        # 모델 로드는 필요할 때 지연 로드
        self.inpaint_pipe = None
        self.img2img_pipe = None
        
        print(f"헤어스타일 생성기 초기화 완료 (장치: {self.device})")
    
    def load_inpaint_model(self):
        """인페인팅 모델 로드"""
        if self.inpaint_pipe is None:
            print(f"인페인팅 모델 로드 중... ({self.inpaint_model_id})")
            
            # 안전 검사기 비활성화 (필요한 경우)
            dummy_safety_checker = lambda images, **kwargs: (images, False)
            
            self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.inpaint_model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=dummy_safety_checker
            )
            self.inpaint_pipe = self.inpaint_pipe.to(self.device)
            print("인페인팅 모델 로드 완료")
        return self.inpaint_pipe
    
    def load_img2img_model(self):
        """이미지-이미지 모델 로드"""
        if self.img2img_pipe is None:
            print(f"이미지-이미지 모델 로드 중... ({self.img2img_model_id})")
            
            # 안전 검사기 비활성화 (필요한 경우)
            dummy_safety_checker = lambda images, **kwargs: (images, False)
            
            self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.img2img_model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=dummy_safety_checker
            )
            self.img2img_pipe = self.img2img_pipe.to(self.device)
            print("이미지-이미지 모델 로드 완료")
        return self.img2img_pipe
    
    def load_reference_images(self, reference_dir_or_files: Union[str, List[str]]) -> List[Image.Image]:
        """
        참조 이미지 불러오기
        
        Args:
            reference_dir_or_files: 참조 이미지 디렉토리 또는 파일 경로 목록
            
        Returns:
            참조 이미지 목록
        """
        reference_images = []
        
        if isinstance(reference_dir_or_files, str):
            # 디렉토리인 경우
            if os.path.isdir(reference_dir_or_files):
                image_paths = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                    image_paths.extend(glob.glob(os.path.join(reference_dir_or_files, ext)))
                    image_paths.extend(glob.glob(os.path.join(reference_dir_or_files, ext.upper())))
                
                print(f"{len(image_paths)}개의 참조 이미지를 찾았습니다.")
                
                for img_path in image_paths:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        reference_images.append(img)
                        print(f"이미지 로드: {img_path}")
                    except Exception as e:
                        print(f"이미지 로드 실패: {img_path}, 오류: {e}")
            else:
                # 단일 파일인 경우
                try:
                    img = Image.open(reference_dir_or_files).convert("RGB")
                    reference_images.append(img)
                    print(f"이미지 로드: {reference_dir_or_files}")
                except Exception as e:
                    print(f"이미지 로드 실패: {reference_dir_or_files}, 오류: {e}")
        else:
            # 파일 경로 목록인 경우
            for img_path in reference_dir_or_files:
                try:
                    img = Image.open(img_path).convert("RGB")
                    reference_images.append(img)
                    print(f"이미지 로드: {img_path}")
                except Exception as e:
                    print(f"이미지 로드 실패: {img_path}, 오류: {e}")
        
        return reference_images
    
    def analyze_reference_images(self, reference_images: List[Image.Image]) -> Tuple[str, Dict]:
        """
        참조 이미지 분석하여 특성 추출
        
        Args:
            reference_images: 참조 이미지 목록
            
        Returns:
            프롬프트와 스타일 특성 사전
        """
        if not reference_images:
            return "", {}
        
        # 기본 스타일 특성
        style_features = {
            "length": "",
            "color": "",
            "texture": "",
            "style": ""
        }
        
        # 간단한 색상 분석
        for img in reference_images:
            # 이미지를 NumPy 배열로 변환
            img_np = np.array(img)
            
            # HSV 색 공간으로 변환
            img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            
            # 전체 이미지 평균 색상 계산
            avg_hue = np.mean(img_hsv[:, :, 0])
            avg_sat = np.mean(img_hsv[:, :, 1])
            avg_val = np.mean(img_hsv[:, :, 2])
            
            # 색상 결정
            # H: 0-180 (OpenCV에서는 0-179)
            if avg_sat < 30:
                if avg_val < 80:
                    color = "black"
                elif avg_val < 160:
                    color = "gray"
                else:
                    color = "white"
            else:
                if avg_hue < 15 or avg_hue > 165:
                    color = "red"
                elif avg_hue < 30:
                    color = "orange"
                elif avg_hue < 45:
                    color = "yellow"
                elif avg_hue < 75:
                    color = "green"
                elif avg_hue < 105:
                    color = "cyan"
                elif avg_hue < 135:
                    color = "blue"
                elif avg_hue < 165:
                    color = "purple"
                else:
                    color = "red"
            
            # 현재 이미지 색상 기록
            if not style_features["color"]:
                style_features["color"] = color
        
        # 일반적인 추정 (실제로는 더 정교한 분석이 필요함)
        # 나중에 더 정교한 머리카락 분석 알고리즘으로 대체 가능
        style_features["length"] = "medium"  # 기본값
        style_features["texture"] = "straight"  # 기본값
        style_features["style"] = "natural"  # 기본값
        
        # 프롬프트 생성
        prompt = f"{style_features['length']} {style_features['texture']} {style_features['color']} hair, {style_features['style']} hairstyle"
        
        # 추가 스타일 용어
        additional_terms = "photo-realistic, professional photography, detailed hair strands, natural lighting"
        
        full_prompt = f"{prompt}, {additional_terms}"
        
        return full_prompt, style_features
    
    def prepare_canvas(self, bald_image_path: str, hair_mask_path: str, face_mask_path: str,
                       target_size: int = 512) -> Tuple[Image.Image, Image.Image, Image.Image, Dict]:
        """
        인페인팅을 위한 캔버스 준비
        
        Args:
            bald_image_path: 대머리 이미지 경로
            hair_mask_path: 머리카락 마스크 경로
            face_mask_path: 얼굴 마스크 경로
            target_size: 모델 입력 크기
            
        Returns:
            (이미지, 마스크, 얼굴 마스크, 크기 정보)
        """
        # 이미지 및 마스크 로드
        bald_image = Image.open(bald_image_path).convert("RGB")
        hair_mask = Image.open(hair_mask_path).convert("L")
        face_mask = Image.open(face_mask_path).convert("L")
        
        # 원본 크기 저장
        original_size = bald_image.size
        
        # 크기 조정
        ratio = min(target_size / original_size[0], target_size / original_size[1])
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        
        resized_image = bald_image.resize(new_size, Image.LANCZOS)
        resized_hair_mask = hair_mask.resize(new_size, Image.NEAREST)
        resized_face_mask = face_mask.resize(new_size, Image.NEAREST)
        
        # 정사각형 캔버스 준비
        square_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        square_hair_mask = Image.new("L", (target_size, target_size), 0)
        square_face_mask = Image.new("L", (target_size, target_size), 0)
        
        # 중앙 배치
        paste_x = (target_size - new_size[0]) // 2
        paste_y = (target_size - new_size[1]) // 2
        
        square_image.paste(resized_image, (paste_x, paste_y))
        square_hair_mask.paste(resized_hair_mask, (paste_x, paste_y))
        square_face_mask.paste(resized_face_mask, (paste_x, paste_y))
        
        # 크기 정보 저장
        size_info = {
            "original_size": original_size,
            "new_size": new_size,
            "paste_x": paste_x,
            "paste_y": paste_y,
            "target_size": target_size,
        }
        
        return square_image, square_hair_mask, square_face_mask, size_info
    
    def save_debug_images(self, output_dir: str, prefix: str, 
                          image: Image.Image, hair_mask: Image.Image,
                          face_mask: Optional[Image.Image] = None,
                          reference_images: Optional[List[Image.Image]] = None):
        """
        디버깅용 이미지 저장
        
        Args:
            output_dir: 출력 디렉토리
            prefix: 파일명 접두사
            image: 입력 이미지
            hair_mask: 머리카락 마스크
            face_mask: 얼굴 마스크 (선택 사항)
            reference_images: 참조 이미지 목록 (선택 사항)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 입력 이미지 저장
        image.save(os.path.join(output_dir, f"{prefix}_input.png"))
        
        # 머리카락 마스크 저장
        hair_mask.save(os.path.join(output_dir, f"{prefix}_hair_mask.png"))
        
        # 얼굴 마스크 저장 (있는 경우)
        if face_mask is not None:
            face_mask.save(os.path.join(output_dir, f"{prefix}_face_mask.png"))
        
        # 참조 이미지 저장 (있는 경우)
        if reference_images:
            for i, ref_img in enumerate(reference_images):
                ref_img.save(os.path.join(output_dir, f"{prefix}_reference_{i+1}.png"))
            
            # 참조 이미지 그리드 생성 (최대 5개까지)
            if len(reference_images) > 1:
                grid_size = min(len(reference_images), 5)
                grid_width = grid_size * 200
                grid_height = 200
                
                grid_img = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))
                
                for i, ref_img in enumerate(reference_images[:grid_size]):
                    # 썸네일 크기로 조정
                    thumb = ref_img.copy()
                    thumb.thumbnail((200, 200))
                    
                    # 그리드에 붙이기
                    paste_x = i * 200 + (200 - thumb.width) // 2
                    paste_y = (200 - thumb.height) // 2
                    grid_img.paste(thumb, (paste_x, paste_y))
                
                grid_img.save(os.path.join(output_dir, f"{prefix}_reference_grid.png"))
    
    def prepare_generation(self, bald_image_path: str, hair_mask_path: str, face_mask_path: str,
                           reference_images_or_dir: Union[str, List[str]], output_dir: str) -> Dict:
        """
        헤어스타일 생성 준비
        
        Args:
            bald_image_path: 대머리 이미지 경로
            hair_mask_path: 머리카락 마스크 경로
            face_mask_path: 얼굴 마스크 경로
            reference_images_or_dir: 참조 이미지 디렉토리 또는 파일 경로 목록
            output_dir: 출력 디렉토리
            
        Returns:
            생성 설정 정보
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 참조 이미지 로드
        reference_images = self.load_reference_images(reference_images_or_dir)
        
        if not reference_images:
            print("경고: 참조 이미지가 로드되지 않았습니다!")
            return None
        
        # 참조 이미지 분석
        prompt, style_features = self.analyze_reference_images(reference_images)
        print(f"생성된 프롬프트: {prompt}")
        print(f"스타일 특성: {style_features}")
        
        # 인페인팅용 캔버스 준비
        canvas_image, hair_mask, face_mask, size_info = self.prepare_canvas(
            bald_image_path, hair_mask_path, face_mask_path
        )
        
        # 디버그 이미지 저장
        self.save_debug_images(
            output_dir, 
            "prepared", 
            canvas_image, 
            hair_mask, 
            face_mask, 
            reference_images
        )
        
        # 생성 설정 정보
        generation_config = {
            "prompt": prompt,
            "style_features": style_features,
            "canvas_path": os.path.join(output_dir, "prepared_input.png"),
            "hair_mask_path": os.path.join(output_dir, "prepared_hair_mask.png"),
            "face_mask_path": os.path.join(output_dir, "prepared_face_mask.png"),
            "reference_grid_path": os.path.join(output_dir, "prepared_reference_grid.png"),
            "size_info": size_info,
            "num_reference_images": len(reference_images),
        }
        
        # 설정 정보 저장
        import json
        with open(os.path.join(output_dir, "generation_config.json"), "w") as f:
            json.dump(generation_config, f, indent=2)
        
        print(f"헤어스타일 생성 준비 완료! 결과가 {output_dir}에 저장되었습니다.")
        return generation_config

def main():
    parser = argparse.ArgumentParser(description="헤어스타일 생성 준비")
    parser.add_argument("--bald", required=True, help="대머리 이미지 경로")
    parser.add_argument("--hair_mask", required=True, help="머리카락 마스크 경로")
    parser.add_argument("--face_mask", required=True, help="얼굴 마스크 경로")
    parser.add_argument("--reference", required=True, nargs='+', help="참조 이미지 경로 또는 디렉토리")
    parser.add_argument("--output", default="hairstyle_preparation", help="출력 디렉토리")
    parser.add_argument("--device", help="장치 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 여러 참조 이미지 경로가 제공된 경우
    reference_images = args.reference
    if len(reference_images) == 1 and os.path.isdir(reference_images[0]):
        # 디렉토리인 경우
        reference_images = reference_images[0]
    
    # 헤어스타일 생성기 초기화
    generator = HairstyleGenerator(device=args.device)
    
    # 생성 준비
    generator.prepare_generation(
        args.bald, 
        args.hair_mask, 
        args.face_mask, 
        reference_images, 
        args.output
    )

if __name__ == "__main__":
    main()
import os
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
import mediapipe as mp
from diffusers import StableDiffusionInpaintPipeline
from insightface.app import FaceAnalysis
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
import glob
from tqdm import tqdm
import json

class ReferenceProcessor:
    """스타일 참조 이미지 처리 클래스"""
    
    def __init__(self):
        """얼굴 감지 및 분석 초기화"""
        # MediaPipe 얼굴 감지
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        
        # InsightFace 초기화
        self.face_analyzer = None
    
    def load_face_analyzer(self):
        """InsightFace 얼굴 분석기 로드"""
        if self.face_analyzer is None:
            try:
                self.face_analyzer = FaceAnalysis(name="buffalo_l")
                self.face_analyzer.prepare(ctx_id=0)
                print("얼굴 분석기 로드 완료")
                return True
            except Exception as e:
                print(f"얼굴 분석기 로드 실패: {e}")
                return False
        return True
    
    def align_and_crop_face(self, image, target_size=512, scale_factor=1.5):
        """얼굴 정렬 및 크롭"""
        # 이미지 변환
        if isinstance(image, Image.Image):
            np_image = np.array(image)
            pil_input = True
        else:
            np_image = image
            pil_input = False
        
        # 얼굴 감지
        rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB) if np_image.shape[-1] == 3 else np_image
        results = self.face_detection.process(rgb_image)
        
        if not results.detections:
            print("얼굴을 찾을 수 없습니다. 중앙 크롭으로 대체합니다.")
            # 중앙 크롭
            h, w = np_image.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            size = min(w, h)
            xmin = max(0, center_x - size // 2)
            ymin = max(0, center_y - size // 2)
            
            cropped = np_image[ymin:ymin+size, xmin:xmin+size]
            resized = cv2.resize(cropped, (target_size, target_size))
            
            if pil_input:
                return Image.fromarray(resized)
            return resized
        
        # 가장 큰 얼굴 선택
        detection = results.detections[0]
        for d in results.detections[1:]:
            if (d.location_data.relative_bounding_box.width * 
                d.location_data.relative_bounding_box.height > 
                detection.location_data.relative_bounding_box.width * 
                detection.location_data.relative_bounding_box.height):
                detection = d
        
        # 얼굴 경계 상자
        bbox = detection.location_data.relative_bounding_box
        h, w = rgb_image.shape[:2]
        
        xmin = max(0, int(bbox.xmin * w))
        ymin = max(0, int(bbox.ymin * h))
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # 확장된 경계 상자 계산
        center_x = xmin + width // 2
        center_y = ymin + height // 2
        
        new_size = int(max(width, height) * scale_factor)
        
        # 경계 확인
        new_xmin = max(0, center_x - new_size // 2)
        new_ymin = max(0, center_y - new_size // 2)
        new_xmax = min(w, new_xmin + new_size)
        new_ymax = min(h, new_ymin + new_size)
        
        # 크롭
        cropped = np_image[new_ymin:new_ymax, new_xmin:new_xmax]
        
        # 패딩 (필요한 경우)
        pad_left, pad_top = 0, 0
        pad_right, pad_bottom = 0, 0
        
        if new_xmin == 0:
            pad_left = new_size - cropped.shape[1]
        if new_ymin == 0:
            pad_top = new_size - cropped.shape[0]
        if new_xmax == w:
            pad_right = new_size - cropped.shape[1]
        if new_ymax == h:
            pad_bottom = new_size - cropped.shape[0]
        
        if pad_left or pad_top or pad_right or pad_bottom:
            cropped = cv2.copyMakeBorder(
                cropped, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        
        # 최종 크기로 리사이즈
        resized = cv2.resize(cropped, (target_size, target_size))
        
        if pil_input:
            return Image.fromarray(resized)
        return resized
    
    def extract_face_embedding(self, image):
        """이미지에서 얼굴 임베딩 추출"""
        # InsightFace 로드 확인
        if not self.load_face_analyzer():
            return None
        
        try:
            # PIL -> OpenCV
            if isinstance(image, Image.Image):
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                cv_image = image.copy()
            
            # 얼굴 감지
            faces = self.face_analyzer.get(cv_image)
            
            if not faces:
                print("얼굴을 찾을 수 없습니다.")
                return None
            
            # 첫 번째(가장 큰) 얼굴만 사용
            face_embedding = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
            return face_embedding
            
        except Exception as e:
            print(f"얼굴 임베딩 추출 오류: {e}")
            return None
    
    def process_reference_directory(self, reference_dir, output_dir, target_size=512):
        """참조 이미지 디렉토리 처리"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 처리 결과 저장
        reference_data = {}
        
        # 이미지 파일 찾기
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(reference_dir, ext)))
            image_files.extend(glob.glob(os.path.join(reference_dir, ext.upper())))
        
        if not image_files:
            print(f"경고: {reference_dir}에서 이미지를 찾을 수 없습니다.")
            return reference_data
        
        print(f"{len(image_files)}개 참조 이미지 처리 중...")
        
        for idx, img_path in enumerate(tqdm(image_files)):
            try:
                # 이미지 로드
                image = Image.open(img_path).convert("RGB")
                
                # 파일명
                filename = os.path.basename(img_path)
                
                # 얼굴 정렬 및 크롭
                processed_image = self.align_and_crop_face(image, target_size)
                
                # 얼굴 임베딩 추출
                face_embedding = self.extract_face_embedding(processed_image)
                
                if face_embedding is None:
                    print(f"이미지에서 얼굴을 찾을 수 없음: {img_path}")
                    continue
                
                # 처리된 이미지 저장
                output_filename = f"ref_{idx+1}_{filename}"
                output_path = os.path.join(output_dir, output_filename)
                processed_image.save(output_path)
                
                # 임베딩 저장 (텐서를 리스트로 변환)
                embedding_np = face_embedding.cpu().numpy()
                
                # 임베딩 정보 저장
                reference_data[output_filename] = {
                    "original_path": img_path,
                    "processed_path": output_path,
                    "has_embedding": True
                }
                
                # 임베딩 파일 별도 저장 (NumPy 배열)
                embedding_path = os.path.join(output_dir, f"embedding_{idx+1}_{filename}.npy")
                np.save(embedding_path, embedding_np)
                
                print(f"처리 완료: {img_path} -> {output_path}")
                
            except Exception as e:
                print(f"이미지 처리 실패 ({img_path}): {e}")
        
        # 참조 정보 JSON으로 저장
        json_path = os.path.join(output_dir, "reference_info.json")
        with open(json_path, 'w') as f:
            json.dump(reference_data, f, indent=2)
        
        return reference_data
    
    def create_reference_grid(self, processed_dir, output_path, max_images=12, 
                           cols=4, thumb_size=256):
        """참조 이미지 그리드 생성"""
        # 이미지 파일 찾기
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(processed_dir, f"ref_*{ext}")))
        
        if not image_files:
            print(f"경고: {processed_dir}에서 처리된 참조 이미지를 찾을 수 없습니다.")
            return None
        
        # 최대 이미지 수 제한
        image_files = image_files[:max_images]
        
        # 행 수 계산
        rows = (len(image_files) + cols - 1) // cols
        
        # 그리드 생성
        grid = Image.new('RGB', (cols * thumb_size, rows * thumb_size), (255, 255, 255))
        
        # 이미지 배치
        for i, img_path in enumerate(image_files):
            try:
                # 이미지 로드 및 썸네일 생성
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((thumb_size, thumb_size))
                
                # 그리드에 배치
                row = i // cols
                col = i % cols
                
                # 중앙 정렬
                x = col * thumb_size + (thumb_size - img.width) // 2
                y = row * thumb_size + (thumb_size - img.height) // 2
                
                grid.paste(img, (x, y))
                
            except Exception as e:
                print(f"이미지 그리드 생성 오류 ({img_path}): {e}")
        
        # 그리드 저장
        grid.save(output_path)
        print(f"이미지 그리드 저장됨: {output_path}")
        
        return output_path


class HairstyleGenerator:
    """참조 이미지를 활용한 헤어스타일 생성"""
    
    def __init__(self, bin_path="models/ip-adapter-faceid-plusv2_sd15.bin", 
               lora_path="models/ip-adapter-faceid-plusv2_sd15_lora.safetensors",
               device=None):
        """생성기 초기화"""
        # 장치 설정
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.bin_path = bin_path
        self.lora_path = lora_path
        
        # 모델 초기화
        self.pipe = None
        self.ip_model = None
        
        # 참조 처리기
        self.reference_processor = ReferenceProcessor()
    
    def load_models(self):
        """모델 로드"""
        print("모델 로드 중...")
        
        try:
            # SD 1.5 인페인팅 모델 로드
            model_path = "stable-diffusion-v1-5/stable-diffusion-inpainting"
            
            # 파이프라인 로드
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                safety_checker=None
            )
            self.pipe = self.pipe.to(self.device)
            
            # 메모리 최적화
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
            
            # IP-Adapter-FaceID-PlusV2 로드
            self.ip_model = IPAdapterFaceIDPlus(
                self.pipe,
                self.bin_path,
                self.lora_path,
                device=self.device
            )
            
            print("모델 로드 완료")
            return True
            
        except Exception as e:
            print(f"모델 로드 오류: {e}")
            return False
    
    def create_hair_mask(self, image, hair_mask_path=None):
        """머리카락 마스크 생성 또는 로드"""
        if hair_mask_path and os.path.exists(hair_mask_path):
            # 기존 마스크 로드
            hair_mask = Image.open(hair_mask_path).convert("L")
            if hair_mask.size != image.size:
                hair_mask = hair_mask.resize(image.size, Image.NEAREST)
            return hair_mask
        
        # 마스크가 없는 경우 - 간단한 기본 마스크 생성
        np_image = np.array(image)
        h, w = np_image.shape[:2]
        
        # 간단한 타원형 마스크 생성 (머리카락 영역 근사값)
        mask = np.zeros((h, w), dtype=np.uint8)
        center_x, center_y = w // 2, h // 3  # 머리는 상단 1/3에 위치
        axis_x, axis_y = w // 2, h // 3      # 타원 크기
        
        cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 
                   0, 0, 360, 255, -1)
        
        # 얼굴 영역 제외 (대략적인 위치)
        face_center_y = h // 2
        face_height = h // 3
        face_width = w // 3
        
        cv2.rectangle(mask, 
                     (center_x - face_width//2, face_center_y - face_height//2),
                     (center_x + face_width//2, face_center_y + face_height//2),
                     0, -1)
        
        return Image.fromarray(mask)
    
    def generate_from_references(self, input_image_path, reference_dir, output_dir="output", 
                              hair_mask_path=None, align_face=True, num_per_reference=1, 
                              prompt=None, negative_prompt=None, num_inference_steps=30, 
                              guidance_scale=7.5, s_scale=0.7, shortcut=True, seed=None):
        """참조 이미지 디렉토리에서 헤어스타일 생성"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 모델 로드 확인
        if self.ip_model is None:
            if not self.load_models():
                raise RuntimeError("모델 로드에 실패했습니다.")
        
        # 입력 이미지 전처리
        print("입력 이미지 전처리 중...")
        input_image = Image.open(input_image_path).convert("RGB")
        
        if align_face:
            input_image = self.reference_processor.align_and_crop_face(input_image)
        
        input_image_path = os.path.join(output_dir, "processed_input.png")
        input_image.save(input_image_path)
        
        # 머리카락 마스크 생성/로드
        print("머리카락 마스크 준비 중...")
        hair_mask = self.create_hair_mask(input_image, hair_mask_path)
        hair_mask_path = os.path.join(output_dir, "hair_mask.png")
        hair_mask.save(hair_mask_path)
        
        # 마스크 준비 (흰색 = 인페인팅 영역)
        hair_mask_np = np.array(hair_mask)
        if hair_mask_np.max() <= 1:  # 0-1 범위인 경우
            hair_mask_np = hair_mask_np * 255
        
        # 마스크 처리: 흰색이 인페인팅 영역이 되도록
        hair_mask_processed = Image.fromarray(hair_mask_np.astype(np.uint8))
        
        # 참조 이미지 디렉토리 찾기
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        reference_files = []
        
        for ext in image_extensions:
            reference_files.extend(glob.glob(os.path.join(reference_dir, ext)))
            reference_files.extend(glob.glob(os.path.join(reference_dir, ext.upper())))
        
        if not reference_files:
            print(f"경고: {reference_dir}에서 참조 이미지를 찾을 수 없습니다.")
            return []
        
        print(f"{len(reference_files)}개 참조 이미지에서 생성 시작...")
        
        # 기본 프롬프트 설정
        if prompt is None:
            prompt = "professional portrait photo of a person, high quality, photorealistic, detailed hair"
        
        if negative_prompt is None:
            negative_prompt = "deformed, distorted, disfigured, bad anatomy, bad proportions, unrealistic, low quality, blurry"
        
        all_results = []
        
        # 참조 이미지별 생성
        for idx, ref_path in enumerate(reference_files):
            try:
                print(f"\n처리 중: 참조 이미지 {idx+1}/{len(reference_files)} - {ref_path}")
                
                # 참조 이미지 로드 및 전처리
                ref_image = Image.open(ref_path).convert("RGB")
                
                if align_face:
                    ref_image = self.reference_processor.align_and_crop_face(ref_image)
                
                # 얼굴 임베딩 추출
                face_embedding = self.reference_processor.extract_face_embedding(ref_image)
                
                if face_embedding is None:
                    print(f"참조 이미지에서 얼굴을 찾을 수 없음: {ref_path}")
                    continue
                
                # 참조 이미지 저장
                ref_filename = os.path.basename(ref_path)
                ref_processed_path = os.path.join(output_dir, f"ref_{idx+1}_{ref_filename}")
                ref_image.save(ref_processed_path)
                
                # 참조 이미지별 여러 샘플 생성
                ref_results = []
                
                for sample_idx in range(num_per_reference):
                    # 샘플마다 다른 시드
                    if seed is None:
                        sample_seed = torch.randint(0, 2**32, (1,)).item()
                        sample_generator = torch.Generator(device=self.device).manual_seed(sample_seed)
                    else:
                        sample_seed = seed + idx * 100 + sample_idx
                        sample_generator = torch.Generator(device=self.device).manual_seed(sample_seed)
                    
                    # 생성
                    try:
                        images = self.ip_model.generate(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=input_image,
                            mask_image=hair_mask_processed,
                            faceid_embeds=face_embedding,
                            num_samples=1,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=sample_generator,
                            s_scale=s_scale,
                            shortcut=shortcut
                        )
                        
                        generated_image = images[0]
                        
                        # 결과 저장
                        result_filename = f"result_ref{idx+1}_sample{sample_idx+1}.png"
                        result_path = os.path.join(output_dir, result_filename)
                        generated_image.save(result_path)
                        
                        ref_results.append(result_path)
                        all_results.append(result_path)
                        
                        print(f"  샘플 {sample_idx+1}/{num_per_reference} 생성 완료 (시드: {sample_seed})")
                        
                    except Exception as e:
                        print(f"  샘플 생성 오류: {e}")
                
                # 참조 이미지별 그리드 생성 (여러 샘플이 있는 경우)
                if len(ref_results) > 1:
                    ref_images = [Image.open(path) for path in ref_results]
                    grid = self._create_image_grid(ref_images)
                    
                    grid_path = os.path.join(output_dir, f"grid_ref{idx+1}.png")
                    grid.save(grid_path)
                
            except Exception as e:
                print(f"참조 이미지 처리 오류 ({ref_path}): {e}")
        
        # 전체 결과 그리드 생성 (모든 참조 이미지에서 하나씩)
        if len(all_results) > 1:
            # 각 참조 이미지당 첫 번째 결과만 사용
            unique_refs = {}
            for path in all_results:
                ref_idx = int(os.path.basename(path).split('_')[1][3:])
                if ref_idx not in unique_refs:
                    unique_refs[ref_idx] = path
            
            grid_images = [Image.open(path) for path in unique_refs.values()]
            if grid_images:
                grid = self._create_image_grid(grid_images)
                
                grid_path = os.path.join(output_dir, "all_results_grid.png")
                grid.save(grid_path)
        
        print(f"\n생성 완료: {len(all_results)}개 이미지가 {output_dir}에 저장되었습니다.")
        return all_results
    
    def _create_image_grid(self, images, rows=1):
        """이미지 그리드 생성"""
        if not images:
            return None
        
        # 그리드 생성
        w, h = images[0].size
        grid_w = len(images) if rows == 1 else len(images) // rows
        grid_h = rows
        
        grid = Image.new('RGB', (w * grid_w, h * grid_h))
        
        for i, img in enumerate(images):
            row = i // grid_w
            col = i % grid_w
            grid.paste(img, (col * w, row * h))
        
        return grid


def main():
    parser = argparse.ArgumentParser(description="참조 이미지 기반 헤어스타일 생성")
    parser.add_argument("--input", required=True, help="입력 이미지 경로")
    parser.add_argument("--reference_dir", required=True, help="참조 이미지 디렉토리")
    parser.add_argument("--output", default="hairstyle_results", help="출력 디렉토리")
    parser.add_argument("--bin_path", default="models/ip-adapter-faceid-plusv2_sdxl.bin", help="IP-Adapter bin 파일 경로")
    parser.add_argument("--lora_path", default="models/ip-adapter-faceid-plusv2_sdxl_lora.safetensors", help="IP-Adapter lora 파일 경로")
    parser.add_argument("--hair_mask", help="머리카락 마스크 경로 (없으면 자동 생성)")
    parser.add_argument("--no_align", action="store_true", help="얼굴 정렬 비활성화")
    parser.add_argument("--samples_per_ref", type=int, default=1, help="참조 이미지당 생성할 샘플 수")
    parser.add_argument("--prompt", help="텍스트 프롬프트 (옵션)")
    parser.add_argument("--steps", type=int, default=30, help="추론 단계 수")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="가이던스 강도")
    parser.add_argument("--s_scale", type=float, default=0.7, help="구조 유지 강도 (0.5-1.0)")
    parser.add_argument("--no_shortcut", action="store_true", help="shortcut 비활성화")
    parser.add_argument("--seed", type=int, help="랜덤 시드")
    parser.add_argument("--process_only", action="store_true", help="참조 이미지만 처리 (생성 없음)")
    
    args = parser.parse_args()
    
    # 1. 참조 이미지 처리만 수행
    if args.process_only:
        processor = ReferenceProcessor()
        processed_dir = os.path.join(args.output, "processed_references")
        processor.process_reference_directory(args.reference_dir, processed_dir)
        
        # 그리드 생성
        processor.create_reference_grid(processed_dir, os.path.join(args.output, "reference_grid.jpg"))
        
    # 2. 전체 파이프라인 실행
    else:
        # 생성기 초기화
        generator = HairstyleGenerator(args.bin_path, args.lora_path)
        
        # 생성 실행
        generator.generate_from_references(
            input_image_path=args.input,
            reference_dir=args.reference_dir,
            output_dir=args.output,
            hair_mask_path=args.hair_mask,
            align_face=not args.no_align,
            num_per_reference=args.samples_per_ref,
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            s_scale=args.s_scale,
            shortcut=not args.no_shortcut,
            seed=args.seed
        )

if __name__ == "__main__":
    main()
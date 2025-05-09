import os
import torch
import numpy as np
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlusXL
from ip_adapter.ip_adapter import IPAdapterPlusXL  # 실제 SDXL용 IP-Adapter import 필요
from PIL import Image

class IPAdapterSDXL:
    def __init__(self, pipe, 
                 faceid_bin_path="models/ip-adapter-faceid-plusv2_sdxl.bin", 
                 faceid_lora_path="models/ip-adapter-faceid-plusv2_sdxl_lora.safetensors",
                 style_bin_path="models/ip-adapter-plus_sdxl_vit-h.bin",
                 device='cuda'):
        self.pipe = pipe
        self.device = device

        clip_model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

        # 1. FaceID 어댑터 초기화
        print("FaceID 어댑터 초기화 중...")
        self.faceid_adapter = IPAdapterFaceIDPlusXL(
            self.pipe,
            clip_model_id,
            faceid_bin_path,
            device=self.device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )

         # 2. 스타일 어댑터 초기화 (헤어스타일 추출용)
        print("스타일 어댑터 초기화 중...")
        self.style_adapter = IPAdapterPlusXL(
            self.pipe,
            clip_model_id,
            style_bin_path,
            device=self.device,
            num_tokens=16
        )

        print("두 어댑터 초기화 완료")

    # def get_faceid_embeds(self, faceid_embeds, face_image, s_scale, shortcut):
    #     """
    #     얼굴 임베딩을 IP-Adapter 이미지 임베딩으로 변환
    #     """
        
    #     # 2. IP-Adapter를 통해 이미지 임베딩으로 변환
    #     if isinstance(face_image, np.ndarray):
    #         face_image = Image.fromarray(face_image)
            
    #     image_prompt_embeds, uncond_image_prompt_embeds = self.faceid_adapter.get_image_embeds(
    #         faceid_embeds=faceid_embeds,
    #         face_image=face_image,
    #         s_scale=s_scale,
    #         shortcut=shortcut
    #     )
        
    #     return image_prompt_embeds, uncond_image_prompt_embeds

    # def get_style_embeds(self, style_images):
    #     """
    #     여러 장의 스타일 이미지를 받아 헤어스타일 임베딩 추출 및 평균
    #     """
    #     if isinstance(style_images, list) and len(style_images) == 0:
    #         return None
            
    #     if not isinstance(style_images, list):
    #         style_images = [style_images]
            
    #     # 각 이미지에서 임베딩 추출
    #     all_embeds = []
    #     for img in style_images:
    #         if isinstance(img, np.ndarray):
    #             img = Image.fromarray(img)
            
    #         # 스타일 어댑터를 사용하여 임베딩 추출
    #         embed = self.style_adapter.get_image_embeds(img)
    #         all_embeds.append(embed)
            
    #     # 여러 스타일 이미지의 임베딩 평균
    #     if len(all_embeds) > 1:
    #         style_embeds = torch.cat(all_embeds).mean(dim=0, keepdim=True)
    #     else:
    #         style_embeds = all_embeds[0]
            
    #     print(f"스타일 임베딩 추출 완료: {style_embeds.shape}")
    #     return style_embeds

    def generate(self, prompt, negative_prompt, face_image, faceid_embeds, style_images, 
                mask_image=None, num_samples=1, num_inference_steps=30, guidance_scale=7.5, 
                faceid_scale=0.8, s_scale=0.5, shortcut=True):
        """
        두 어댑터를 조합하여 이미지 생성
        - faceid_scale: 얼굴 특징을 얼마나 강하게 반영할지 결정 (0.0-1.0)
        - style_scale: 헤어스타일 등 스타일 특징을 얼마나 강하게 반영할지 결정 (0.0-1.0)
        """
        # 생성에 필요한 seed 설정
        generator = torch.Generator(device=self.device)
        if seed := torch.randint(0, 2**32, (1,)).item():
            generator = generator.manual_seed(seed)
            
        # 기본 파이프라인 설정
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        self.pipe.set_ip_adapter_scale([0.7, 0.3])  # FaceID 70%, Style 30%
        
        # 모델 생성 과정 실행 
        images = self.pipe(
            prompt=prompt,
            # face_image=face_image,
            faceid_embeds=faceid_embeds,
            negative_prompt=negative_prompt,
            ip_adapter_image=style_images,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            mask_image=mask_image,
            s_scale=s_scale,
        ).images
        
        print(f"이미지 생성 완료: {len(images)}개 생성됨 (시드: {seed})")
        return images
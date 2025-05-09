import os
from PIL import Image, ImageOps
import numpy as np
import torch
from modules.reference_processor_sdxl import ReferenceProcessorSDXL
from modules.ip_adapter_sdxl import IPAdapterSDXL
from diffusers import AutoPipelineForText2Image , DPMSolverMultistepScheduler
from transformers import CLIPVisionModelWithProjection

class HairstyleGeneratorSDXL:
    def __init__(self, faceid_bin_path="models/ip-adapter-faceid-plusv2_sdxl.bin", 
                 faceid_lora_path="models/ip-adapter-faceid-plusv2_sdxl_lora.safetensors",
                 style_bin_path="models/ip-adapter-plus_sdxl_vit-h.bin", device='cuda'):
         
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", 
            subfolder="sdxl_models/image_encoder",
            torch_dtype=torch.float16
        ).to(device)

        self.device = device
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            # "SG161222/RealVisXL_V3.0",
            # "SG161222/RealVisXL_V5.0",
            "stabilityai/stable-diffusion-xl-base-1.0",
            image_encoder=self.image_encoder,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        # self.pipe.load_ip_adapter(
        #     "h94/IP-Adapter",  # 모든 어댑터는 동일 저장소에서 로드
        #     subfolder="sdxl_models",  # SDXL 전용 서브폴더
        #     weight_name="ip-adapter-plus_sdxl_vit-h.safetensors"
        # )
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",  # 모든 어댑터는 동일 저장소에서 로드
            subfolder="sdxl_models",  # SDXL 전용 서브폴더
            weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors"
        )
        # self.pipe.set_ip_adapter_scale([0.7, 0.3])  # FaceID 70%, Style 30%

        self.reference_processor = ReferenceProcessorSDXL()


    def generate_from_references(self, input_image_path, reference_dir, output_dir="output",
                                align_face=True, num_per_reference=1,
                                prompt=None, negative_prompt=None, num_inference_steps=30,
                                guidance_scale=7.5, s_scale=0.7, shortcut=True, seed=None):
        os.makedirs(output_dir, exist_ok=True)
        # 입력 이미지 전처리
        input_image = Image.open(input_image_path).convert("RGB")
        if align_face:
            input_image, faceid_embeds = self.reference_processor.align_and_crop_face_with_insightface(input_image, target_size=1024)
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        input_image_path = os.path.join(output_dir, "processed_input.png")
        input_image.save(input_image_path)

        # # 얼굴 ID 임베딩 추출
        # print("얼굴 ID 임베딩 추출 중...")
        # faceid_embeds = self.ip_adapter.get_faceid_embeds(faceid_embeds, input_image, s_scale, shortcut)

       # 참조 이미지 로드 및 스타일 임베딩 추출
        reference_files = [os.path.join(reference_dir, f) for f in os.listdir(reference_dir)
                          if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
        
        if not reference_files:
            print(f"경고: {reference_dir}에서 참조 이미지를 찾을 수 없습니다.")
            return []
            
        print(f"{len(reference_files)}개 참조 이미지에서 헤어스타일 임베딩 추출 중...")
        # style_images = [Image.open(f).convert("RGB") for f in reference_files]
        style_images = self.reference_processor.process_reference_directory(reference_dir, output_dir, target_size=224, scale_factor=1.7)
        # # 스타일 이미지 저장 (디버깅용)
        # for i, img in enumerate(style_images):
        #     img.save(os.path.join(output_dir, f"reference_{i+1}.png"))

        input_image_resized = input_image.resize((224, 224), Image.BICUBIC)
        # input_tensor = torch.from_numpy(np.array(input_image_resized)).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255.0
        # output = self.pipe.image_encoder(input_tensor)
        # image_embeds = output.image_embeds
        # print("image_embeds shape:", image_embeds.shape, "dtype:", image_embeds.dtype)
        all_results = []
        for sample_idx in range(num_per_reference):
            sample_seed = seed + sample_idx if seed is not None else torch.randint(0, 2**32, (1,)).item()
            sample_generator = torch.Generator(device=self.device).manual_seed(sample_seed)
            print("style_images 타입:", type(style_images))
            print("style_images[0] 타입:", type(style_images[0]))
            print("input_image_resized 타입:", type(input_image_resized))
            print("ip_adapter_image 구조:", [type(x) for x in [style_images, input_image_resized]])
            images = self.pipe(
                prompt=prompt or "professional portrait photo of a person, high quality, photorealistic, detailed hair, 4k",
                negative_prompt=negative_prompt or "deformed, distorted, disfigured, bad anatomy, bad proportions, unrealistic, low quality, blurry",
                ip_adapter_image=input_image_resized,
                # style_images=style_images,
                num_samples=1,
                generator=sample_generator, 
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                s_scale=s_scale,
                shortcut=shortcut
            )
            generated_image = images[0]
            result_filename = f"result_sample{sample_idx+1}.png"
            result_path = os.path.join(output_dir, result_filename)
            generated_image.save(result_path)
            all_results.append(result_path)
            print(f"  샘플 {sample_idx+1}/{num_per_reference} 생성 완료 (시드: {sample_seed})")
        print(f"\n생성 완료: {len(all_results)}개 이미지가 {output_dir}에 저장되었습니다.")
        return all_results

    def _create_image_grid(self, images, rows=1):
        if not images:
            return None
        w, h = images[0].size
        grid_w = len(images) if rows == 1 else len(images) // rows
        grid_h = rows
        grid = Image.new('RGB', (w * grid_w, h * grid_h))
        for i, img in enumerate(images):
            row = i // grid_w
            col = i % grid_w
            grid.paste(img, (col * w, row * h))
        return grid 
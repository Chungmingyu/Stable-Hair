import os
import glob
import cv2
from PIL import Image
import numpy as np
import torch
import mediapipe as mp
from insightface.app import FaceAnalysis

class ReferenceProcessorSDXL:
    def __init__(self):
        self.face_analyzer = None

    def load_face_analyzer(self):
        if self.face_analyzer is None:
            self.face_analyzer = FaceAnalysis(name="buffalo_l")
            self.face_analyzer.prepare(ctx_id=0)
        return self.face_analyzer

    def align_and_crop_face_with_insightface(self, cv_image, target_size=1024, scale_factor=1.5):
        self.load_face_analyzer()
        if cv_image is None:
            print("cv_image is None!")
            return None
        cv_image = np.array(cv_image)
        if cv_image.shape[-1] == 4:
            cv_image = cv_image[..., :3]
        if cv_image.dtype != np.uint8:
            cv_image = (cv_image * 255).clip(0, 255).astype(np.uint8)
        print("InsightFace 입력 shape:", cv_image.shape, "dtype:", cv_image.dtype, "min:", cv_image.min(), "max:", cv_image.max())
        faces = self.face_analyzer.get(cv_image)
        print("InsightFace 감지된 얼굴 수:", len(faces))
        if not faces:
            print("InsightFace로 얼굴 감지 실패!")
            return None
        bbox = faces[0].bbox.astype(int)
        x1, y1, x2, y2 = bbox
        cropped = cv_image[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (target_size, target_size))
        if not faces:
            print("extract_face_embedding: 얼굴을 찾을 수 없습니다.")
            return None
        face_embedding = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        return resized, face_embedding

    def process_reference_directory(self, reference_dir, output_dir, target_size=224, scale_factor=1.7):
        os.makedirs(output_dir, exist_ok=True)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(reference_dir, ext)))
            image_files.extend(glob.glob(os.path.join(reference_dir, ext.upper())))
        aligned_images = []
        for idx, img_path in enumerate(image_files):
            try:
                cv_image = cv2.imread(img_path)
                if cv_image is None:
                    print(f"이미지 로드 실패: {img_path}")
                    continue
                filename = os.path.basename(img_path)
                self.load_face_analyzer()
                faces = self.face_analyzer.get(cv_image)
                print("InsightFace 감지된 얼굴 수:", len(faces))
                if not faces:
                    print(f"InsightFace로 얼굴 감지 실패: {img_path}")
                    continue
                bbox = faces[0].bbox.astype(int)
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                nx1 = max(0, cx - new_w // 2)
                ny1 = max(0, cy - new_h // 2)
                nx2 = min(cv_image.shape[1], cx + new_w // 2)
                ny2 = min(cv_image.shape[0], cy + new_h // 2)
                cropped = cv_image[ny1:ny2, nx1:nx2]
                resized = cv2.resize(cropped, (target_size, target_size))
                pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                output_filename = f"ref_{idx+1}_{filename}"
                output_path = os.path.join(output_dir, output_filename)
                pil_img.save(output_path)
                print(f"처리 완료: {img_path} -> {output_path}")
                aligned_images.append(pil_img)
            except Exception as e:
                print(f"이미지 처리 실패 ({img_path}): {e}")
        return aligned_images

    def create_reference_grid(self, processed_dir, output_path, max_images=12, cols=4, thumb_size=256):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(processed_dir, f"ref_*{ext}")))
        image_files = image_files[:max_images]
        rows = (len(image_files) + cols - 1) // cols
        grid = np.ones((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8) * 255
        for i, img_path in enumerate(image_files):
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (thumb_size, thumb_size))
                row = i // cols
                col = i % cols
                y = row * thumb_size
                x = col * thumb_size
                grid[y:y+img.shape[0], x:x+img.shape[1], :] = img
            except Exception as e:
                print(f"이미지 그리드 생성 오류 ({img_path}): {e}")
        grid_img = Image.fromarray(grid)
        grid_img.save(output_path)
        print(f"이미지 그리드 저장됨: {output_path}")
        return output_path 
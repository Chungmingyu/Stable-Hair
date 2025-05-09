import cv2
import numpy as np
import mediapipe as mp
import os

class FaceProcessor:
    """
    헤어스타일 변경을 위한 간소화된 얼굴 처리 모듈
    - 얼굴 정렬 수행 (1도 이상 기울어진 경우)
    - 512x512 크기로 리사이징
    """
    
    def __init__(self):
        """
        얼굴 처리 모듈 초기화
        """
        # MediaPipe Face Mesh 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # MediaPipe Face Detection (얼굴 감지 실패 시 대비)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0: 근거리, 1: 원거리
            min_detection_confidence=0.5
        )
    
    def align_face(self, image, angle_threshold=1.0):
        """
        이미지에서 얼굴을 찾아 수평으로 정렬
        
        Args:
            image: 입력 이미지 (numpy 배열, BGR 형식)
            angle_threshold: 회전을 적용할 각도 임계값 (기본값: 1.0도)
            
        Returns:
            정렬된 이미지 (BGR 형식)
        """
        # BGR을 RGB로 변환
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 원본 크기 저장
        h, w = rgb_img.shape[:2]
        
        # 얼굴 감지 및 랜드마크 추출
        eyes = self._get_eye_coordinates(rgb_img)
        
        # 눈 좌표를 찾을 수 없으면 원본 이미지 반환
        if not eyes or len(eyes) < 2:
            print("경고: 얼굴을 찾을 수 없습니다. 원본 이미지를 반환합니다.")
            return image
        
        # 눈 위치 기반 회전 각도 계산
        left_eye, right_eye = eyes
        angle = self._calculate_angle(left_eye, right_eye)
        
        # 회전이 필요한지 확인 (임계값 이상 기울어진 경우에만)
        if abs(angle) <= angle_threshold:
            print(f"정렬이 필요하지 않음 (각도: {angle:.2f}도, 임계값: {angle_threshold}도)")
            return image
        
        # 이미지 회전
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        
        # 회전 변환 행렬
        M = cv2.getRotationMatrix2D(eye_center, angle, 1)
        aligned_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        print(f"얼굴 정렬 완료 (각도: {angle:.2f}도)")
        return aligned_img
    
    def _get_eye_coordinates(self, image):
        """
        이미지에서 눈 좌표 추출
        
        Args:
            image: RGB 형식 이미지
            
        Returns:
            눈 좌표 리스트 [left_eye, right_eye] 또는 None
        """
        h, w = image.shape[:2]
        
        # 얼굴 메시 검출
        face_mesh_results = self.face_mesh.process(image)
        if face_mesh_results.multi_face_landmarks:
            # 첫 번째 얼굴의 랜드마크
            face_landmarks = face_mesh_results.multi_face_landmarks[0]
            
            # 왼쪽 눈 (33), 오른쪽 눈 (263) 인덱스
            left_eye = (int(face_landmarks.landmark[33].x * w), 
                       int(face_landmarks.landmark[33].y * h))
            right_eye = (int(face_landmarks.landmark[263].x * w), 
                        int(face_landmarks.landmark[263].y * h))
            
            return [left_eye, right_eye]
        
        # 얼굴 메시가 실패하면 얼굴 검출 시도
        detection_results = self.face_detection.process(image)
        if detection_results.detections:
            # 첫 번째 얼굴 검출 결과
            detection = detection_results.detections[0]
            
            # 눈 랜드마크 추출
            left_eye_idx = 0  # 왼쪽 눈
            right_eye_idx = 1  # 오른쪽 눈
            
            left_eye = (
                int(detection.location_data.relative_keypoints[left_eye_idx].x * w),
                int(detection.location_data.relative_keypoints[left_eye_idx].y * h)
            )
            right_eye = (
                int(detection.location_data.relative_keypoints[right_eye_idx].x * w),
                int(detection.location_data.relative_keypoints[right_eye_idx].y * h)
            )
            
            return [left_eye, right_eye]
        
        # 얼굴 감지 실패
        return None
    
    def _calculate_angle(self, left_eye, right_eye):
        """
        두 눈 위치를 기반으로 회전 각도 계산
        
        Args:
            left_eye: 왼쪽 눈 좌표 (x, y)
            right_eye: 오른쪽 눈 좌표 (x, y)
            
        Returns:
            회전 각도 (도)
        """
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        
        # 각도 계산 (라디안)
        angle = np.arctan2(dy, dx)
        
        # 라디안에서 도로 변환
        angle = np.degrees(angle)
        
        return angle
    
    def resize_to_512(self, image):
        """
        이미지를 512x512 크기로 리사이징
        
        Args:
            image: 입력 이미지 (BGR 형식)
            
        Returns:
            리사이징된 이미지 (BGR 형식)
        """
        return cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    
    def process(self, image_path, output_dir=None):
        """
        이미지 처리 파이프라인 실행
        
        Args:
            image_path: 입력 이미지 경로 또는 이미지 배열
            output_dir: 출력 디렉토리 (None이면 저장하지 않음)
            
        Returns:
            정렬된 이미지, 리사이징된 이미지(512x512), 파일 경로 딕셔너리
        """
        # 이미지 로드
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        else:
            image = image_path  # 이미 배열인 경우
        
        # 1. 얼굴 정렬
        aligned_image = self.align_face(image)
        
        # 2. 512x512로 리사이징
        resized_image = self.resize_to_512(aligned_image)
        
        # 결과 저장 (output_dir이 제공된 경우)
        result_paths = {}
        if output_dir and isinstance(image_path, str):
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # 정렬된 이미지 저장
            aligned_path = os.path.join(output_dir, f"{base_name}_aligned.jpg")
            cv2.imwrite(aligned_path, aligned_image)
            result_paths["aligned"] = aligned_path
            
            # 리사이징된 이미지 저장
            resized_path = os.path.join(output_dir, f"{base_name}_512.jpg")
            cv2.imwrite(resized_path, resized_image)
            result_paths["resized"] = resized_path
        
        return aligned_image, resized_image, result_paths

# 간단한 사용 예시
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print(f"사용법: python {sys.argv[0]} 이미지파일 [출력폴더]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    processor = FaceProcessor()
    aligned, resized, paths = processor.process(image_path, output_dir)
    
    print(f"처리 완료:")
    for key, path in paths.items():
        print(f"- {key}: {path}")
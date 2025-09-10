# face_analyzer.py
import numpy as np
import math

class FaceAnalyzer:
    def __init__(self):
        # MediaPipe 얼굴 랜드마크 주요 포인트 인덱스
        self.FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.NOSE = [1, 2, 5, 6, 19, 20, 94, 168, 195, 197, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 279, 360, 363]
        self.MOUTH = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        
    def calculate_distance(self, point1, point2):
        """두 점 사이의 유클리드 거리 계산"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def analyze_face_features(self, landmarks):
        """랜드마크로부터 얼굴 특징 분석"""
        try:
            # 랜드마크를 좌표 리스트로 변환
            if isinstance(landmarks[0], dict):
                # FastAPI에서 오는 형태: [{"x": 0.5, "y": 0.3, "z": -0.1}, ...]
                points = [(lm["x"], lm["y"]) for lm in landmarks]
            else:
                # MediaPipe 원본 형태
                points = [(lm.x, lm.y) for lm in landmarks]
            
            features = {}
            
            # 1. 얼굴 너비 분석
            features["face_width"] = self._analyze_face_width(points)
            
            # 2. 눈 모양 분석
            features["eye_shape"] = self._analyze_eye_shape(points)
            
            # 3. 코 크기 분석
            features["nose_size"] = self._analyze_nose_size(points)
            
            # 4. 입 크기 분석
            features["mouth_width"] = self._analyze_mouth_width(points)
            
            # 5. 얼굴 길이 분석
            features["face_length"] = self._analyze_face_length(points)
            
            return features
            
        except Exception as e:
            print(f"얼굴 특징 분석 중 오류: {e}")
            return None
    
    def _analyze_face_width(self, points):
        """얼굴 너비 분석"""
        try:
            # 좌우 볼의 가장 넓은 부분
            left_cheek = points[234]   # 왼쪽 볼
            right_cheek = points[454]  # 오른쪽 볼
            face_width = self.calculate_distance(left_cheek, right_cheek)
            
            if face_width > 0.25:
                return "very_wide"
            elif face_width > 0.22:
                return "wide"
            elif face_width > 0.19:
                return "medium"
            elif face_width > 0.16:
                return "narrow"
            else:
                return "very_narrow"
        except:
            return "medium"
    
    def _analyze_eye_shape(self, points):
        """눈 모양 분석"""
        try:
            # 왼쪽 눈의 가로/세로 비율 계산
            left_eye_left = points[33]
            left_eye_right = points[133]
            left_eye_top = points[159]
            left_eye_bottom = points[145]
            
            eye_width = self.calculate_distance(left_eye_left, left_eye_right)
            eye_height = self.calculate_distance(left_eye_top, left_eye_bottom)
            
            # 눈의 가로/세로 비율
            if eye_height == 0:
                ratio = 3  # 기본값
            else:
                ratio = eye_width / eye_height
            
            # 눈 크기도 고려
            eye_area = eye_width * eye_height
            
            if eye_area > 0.003:  # 큰 눈
                return "large"
            elif ratio > 3.5:  # 좁은 눈
                return "narrow"
            elif ratio < 2.5:  # 둥근 눈
                return "round"
            else:
                return "oval"
        except:
            return "oval"
    
    def _analyze_nose_size(self, points):
        """코 크기 분석"""
        try:
            # 코의 너비와 높이 계산
            nose_left = points[220]
            nose_right = points[440]
            nose_top = points[6]
            nose_bottom = points[2]
            
            nose_width = self.calculate_distance(nose_left, nose_right)
            nose_height = self.calculate_distance(nose_top, nose_bottom)
            nose_area = nose_width * nose_height
            
            if nose_area > 0.006:
                return "very_large"
            elif nose_area > 0.004:
                return "large"
            elif nose_area > 0.002:
                return "medium"
            elif nose_area > 0.001:
                return "small"
            else:
                return "very_small"
        except:
            return "medium"
    
    def _analyze_mouth_width(self, points):
        """입 크기 분석"""
        try:
            # 입꼬리 사이의 거리
            mouth_left = points[61]
            mouth_right = points[291]
            mouth_width = self.calculate_distance(mouth_left, mouth_right)
            
            if mouth_width > 0.08:
                return "very_wide"
            elif mouth_width > 0.065:
                return "wide"
            elif mouth_width > 0.05:
                return "medium"
            elif mouth_width > 0.035:
                return "small"
            else:
                return "very_small"
        except:
            return "medium"
    
    def _analyze_face_length(self, points):
        """얼굴 길이 분석"""
        try:
            # 이마부터 턱까지의 거리
            forehead = points[10]  # 이마 중앙
            chin = points[175]     # 턱 끝
            face_length = self.calculate_distance(forehead, chin)
            
            if face_length > 0.35:
                return "very_long"
            elif face_length > 0.3:
                return "long"
            elif face_length > 0.25:
                return "medium"
            elif face_length > 0.2:
                return "short"
            else:
                return "very_short"
        except:
            return "medium"
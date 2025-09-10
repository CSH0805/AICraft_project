# dog_matcher.py
from dog_database import DOG_BREEDS, FEATURE_SCORES

class DogMatcher:
    def __init__(self):
        self.feature_weights = {
            "face_width": 0.25,
            "eye_shape": 0.25,
            "nose_size": 0.2,
            "mouth_width": 0.15,
            "face_length": 0.15
        }
    
    def calculate_similarity(self, human_features, dog_features):
        """사람과 강아지 특징 간 유사도 계산"""
        total_score = 0
        max_possible_score = 0
        
        for feature, weight in self.feature_weights.items():
            if feature in human_features and feature in dog_features:
                human_value = human_features[feature]
                dog_value = dog_features[feature]
                
                # 특징값을 숫자로 변환
                human_score = FEATURE_SCORES.get(feature, {}).get(human_value, 3)
                dog_score = FEATURE_SCORES.get(feature, {}).get(dog_value, 3)
                
                # 차이가 적을수록 높은 점수
                diff = abs(human_score - dog_score)
                similarity = max(0, 5 - diff)  # 0~5 점수
                
                total_score += similarity * weight
                max_possible_score += 5 * weight
        
        # 0~100 퍼센트로 변환
        if max_possible_score > 0:
            return (total_score / max_possible_score) * 100
        return 0
    
    def find_best_matches(self, human_features, top_n=3):
        """가장 닮은 강아지 품종들을 찾기"""
        matches = []
        
        for breed_name, breed_info in DOG_BREEDS.items():
            dog_features = breed_info["face_features"]
            similarity = self.calculate_similarity(human_features, dog_features)
            
            matches.append({
                "breed": breed_name,
                "similarity": similarity,
                "description": breed_info["description"],
                "personality": breed_info["personality"],
                "image": breed_info["image"],
                "matching_features": self._get_matching_features(human_features, dog_features)
            })
        
        # 유사도 순으로 정렬
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches[:top_n]
    
    def _get_matching_features(self, human_features, dog_features):
        """일치하는 특징들 찾기"""
        matching = []
        feature_names = {
            "face_width": "얼굴 너비",
            "eye_shape": "눈 모양",
            "nose_size": "코 크기",
            "mouth_width": "입 크기",
            "face_length": "얼굴 길이"
        }
        
        for feature, human_value in human_features.items():
            if feature in dog_features:
                dog_value = dog_features[feature]
                if human_value == dog_value:
                    matching.append(feature_names.get(feature, feature))
        
        return matching
    
    def get_detailed_analysis(self, human_features):
        """상세한 얼굴 분석 결과 제공"""
        analysis = {
            "face_type": self._determine_face_type(human_features),
            "dominant_features": self._get_dominant_features(human_features),
            "recommendations": self._get_breed_recommendations(human_features)
        }
        return analysis
    
    def _determine_face_type(self, features):
        """얼굴형 판단"""
        width = features.get("face_width", "medium")
        length = features.get("face_length", "medium")
        
        if width in ["wide", "very_wide"] and length == "short":
            return "둥근형"
        elif width in ["narrow", "very_narrow"] and length in ["long", "very_long"]:
            return "긴 얼굴형"
        elif width == "medium" and length == "medium":
            return "표준형"
        elif width in ["wide", "very_wide"]:
            return "넓은 얼굴형"
        else:
            return "독특한 얼굴형"
    
    def _get_dominant_features(self, features):
        """주요 특징 추출"""
        dominant = []
        
        if features.get("eye_shape") == "large":
            dominant.append("큰 눈")
        elif features.get("eye_shape") == "narrow":
            dominant.append("좁은 눈")
        
        if features.get("nose_size") in ["large", "very_large"]:
            dominant.append("큰 코")
        elif features.get("nose_size") in ["small", "very_small"]:
            dominant.append("작은 코")
        
        if features.get("mouth_width") in ["wide", "very_wide"]:
            dominant.append("넓은 입")
        
        return dominant
    
    def _get_breed_recommendations(self, features):
        """특징 기반 품종 추천 이유"""
        recommendations = []
        
        # 얼굴 너비에 따른 추천
        width = features.get("face_width")
        if width in ["wide", "very_wide"]:
            recommendations.append("넓은 얼굴: 골든 리트리버, 불독, 라브라도와 잘 맞습니다")
        elif width in ["narrow", "very_narrow"]:
            recommendations.append("좁은 얼굴: 시바견, 치와와와 유사한 특징입니다")
        
        # 눈 모양에 따른 추천
        eye = features.get("eye_shape")
        if eye == "large":
            recommendations.append("큰 눈: 치와와의 특징적인 큰 눈과 닮았습니다")
        elif eye == "narrow":
            recommendations.append("좁은 눈: 시바견, 허스키의 날카로운 눈매와 유사합니다")
        
        return recommendations
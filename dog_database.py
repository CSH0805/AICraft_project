# dog_database.py
DOG_BREEDS = {
    "골든 리트리버": {
        "name": "골든 리트리버",
        "description": "온순하고 친근한 성격의 대형견",
        "face_features": {
            "face_width": "wide",      # 넓은 얼굴
            "eye_shape": "round",      # 둥근 눈
            "nose_size": "medium",     # 중간 크기 코
            "mouth_width": "wide",     # 넓은 입
            "face_length": "medium"    # 중간 길이 얼굴
        },
        "personality": ["친근함", "온순함", "활발함"],
        "image": "golden_retriever.jpg"
    },
    
    "시바견": {
        "name": "시바견",
        "description": "도도하고 독립적인 성격의 일본 견종",
        "face_features": {
            "face_width": "narrow",    # 좁은 얼굴
            "eye_shape": "narrow",     # 좁은 눈 (여우상)
            "nose_size": "small",      # 작은 코
            "mouth_width": "small",    # 작은 입
            "face_length": "long"      # 긴 얼굴
        },
        "personality": ["도도함", "독립적", "영리함"],
        "image": "shiba_inu.jpg"
    },
    
    "푸들": {
        "name": "푸들",
        "description": "영리하고 우아한 성격의 곱슬모 견종",
        "face_features": {
            "face_width": "medium",    # 중간 너비 얼굴
            "eye_shape": "oval",       # 타원형 눈
            "nose_size": "small",      # 작은 코
            "mouth_width": "small",    # 작은 입
            "face_length": "long"      # 긴 얼굴
        },
        "personality": ["영리함", "우아함", "활발함"],
        "image": "poodle.jpg"
    },
    
    "불독": {
        "name": "불독",
        "description": "묵직하고 차분한 성격의 단두종",
        "face_features": {
            "face_width": "very_wide", # 매우 넓은 얼굴
            "eye_shape": "round",      # 둥근 눈
            "nose_size": "large",      # 큰 코 (낮고 넓음)
            "mouth_width": "wide",     # 넓은 입
            "face_length": "short"     # 짧은 얼굴
        },
        "personality": ["차분함", "묵직함", "충실함"],
        "image": "bulldog.jpg"
    },
    
    "비글": {
        "name": "비글",
        "description": "호기심 많고 활발한 중형 사냥견",
        "face_features": {
            "face_width": "medium",    # 중간 너비
            "eye_shape": "round",      # 둥근 눈
            "nose_size": "medium",     # 중간 크기 코
            "mouth_width": "medium",   # 중간 크기 입
            "face_length": "medium"    # 중간 길이
        },
        "personality": ["호기심", "활발함", "사교적"],
        "image": "beagle.jpg"
    },
    
    "치와와": {
        "name": "치와와",
        "description": "작지만 용감한 초소형 견종",
        "face_features": {
            "face_width": "narrow",    # 좁은 얼굴
            "eye_shape": "large",      # 큰 눈
            "nose_size": "very_small", # 매우 작은 코
            "mouth_width": "small",    # 작은 입
            "face_length": "short"     # 짧은 얼굴
        },
        "personality": ["용감함", "경계심", "애교"],
        "image": "chihuahua.jpg"
    },
    
    "허스키": {
        "name": "시베리안 허스키",
        "description": "늑대 같은 외모의 활동적인 견종",
        "face_features": {
            "face_width": "medium",    # 중간 너비
            "eye_shape": "narrow",     # 좁은 눈 (늑대상)
            "nose_size": "medium",     # 중간 코
            "mouth_width": "medium",   # 중간 입
            "face_length": "long"      # 긴 얼굴
        },
        "personality": ["활동적", "독립적", "친근함"],
        "image": "husky.jpg"
    },
    
    "라브라도": {
        "name": "라브라도 리트리버",
        "description": "충실하고 온화한 대형 가정견",
        "face_features": {
            "face_width": "wide",      # 넓은 얼굴
            "eye_shape": "round",      # 둥근 눈
            "nose_size": "large",      # 큰 코
            "mouth_width": "wide",     # 넓은 입
            "face_length": "medium"    # 중간 길이
        },
        "personality": ["충실함", "온화함", "사교적"],
        "image": "labrador.jpg"
    }
}

# 특징별 점수 매핑
FEATURE_SCORES = {
    "face_width": {
        "very_wide": 5,
        "wide": 4,
        "medium": 3,
        "narrow": 2,
        "very_narrow": 1
    },
    "eye_shape": {
        "large": 5,
        "round": 4,
        "oval": 3,
        "narrow": 2,
        "very_narrow": 1
    },
    "nose_size": {
        "very_large": 5,
        "large": 4,
        "medium": 3,
        "small": 2,
        "very_small": 1
    },
    "mouth_width": {
        "very_wide": 5,
        "wide": 4,
        "medium": 3,
        "small": 2,
        "very_small": 1
    },
    "face_length": {
        "very_long": 5,
        "long": 4,
        "medium": 3,
        "short": 2,
        "very_short": 1
    }
}

def get_dog_info(breed_name):
    """특정 강아지 품종 정보 반환"""
    return DOG_BREEDS.get(breed_name, None)

def get_all_breeds():
    """모든 강아지 품종 리스트 반환"""
    return list(DOG_BREEDS.keys())
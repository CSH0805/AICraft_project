# dog_database.py (업데이트된 버전)
DOG_BREEDS = {
    "골든 리트리버": {
        "name": "골든 리트리버",
        "description": "온순하고 친근한 성격의 대형견",
        "face_features": {
            "face_width": "wide",
            "eye_shape": "round",
            "nose_size": "medium",
            "mouth_width": "wide",
            "face_length": "medium"
        },
        "personality": ["친근함", "온순함", "활발함"],
        "image": "golden_retriever.png"
    },
    
    "시바견": {
        "name": "시바견",
        "description": "도도하고 독립적인 성격의 일본 견종",
        "face_features": {
            "face_width": "narrow",
            "eye_shape": "narrow",
            "nose_size": "small",
            "mouth_width": "small",
            "face_length": "long"
        },
        "personality": ["도도함", "독립적", "영리함"],
        "image": "Shiba_Inu.png"
    },
    
    "푸들": {
        "name": "푸들",
        "description": "영리하고 우아한 성격의 곱슬모 견종",
        "face_features": {
            "face_width": "medium",
            "eye_shape": "oval",
            "nose_size": "small",
            "mouth_width": "small",
            "face_length": "long"
        },
        "personality": ["영리함", "우아함", "활발함"],
        "image": "poodle.png"
    },
    
    "불독": {
        "name": "불독",
        "description": "묵직하고 차분한 성격의 단두종",
        "face_features": {
            "face_width": "very_wide",
            "eye_shape": "round",
            "nose_size": "large",
            "mouth_width": "wide",
            "face_length": "short"
        },
        "personality": ["차분함", "묵직함", "충실함"],
        "image": "bulldog.png"
    },
    
    "비글": {
        "name": "비글",
        "description": "호기심 많고 활발한 중형 사냥견",
        "face_features": {
            "face_width": "medium",
            "eye_shape": "round",
            "nose_size": "medium",
            "mouth_width": "medium",
            "face_length": "medium"
        },
        "personality": ["호기심", "활발함", "사교적"],
        "image": "beagle.png"
    },
    
    "치와와": {
        "name": "치와와",
        "description": "작지만 용감한 초소형 견종",
        "face_features": {
            "face_width": "narrow",
            "eye_shape": "large",
            "nose_size": "very_small",
            "mouth_width": "small",
            "face_length": "short"
        },
        "personality": ["용감함", "경계심", "애교"],
        "image": "chihuahua.png"
    },
    
    "허스키": {
        "name": "시베리안 허스키",
        "description": "늑대 같은 외모의 활동적인 견종",
        "face_features": {
            "face_width": "medium",
            "eye_shape": "narrow",
            "nose_size": "medium",
            "mouth_width": "medium",
            "face_length": "long"
        },
        "personality": ["활동적", "독립적", "친근함"],
        "image": "Siberian_Husky.png"
    },
    
    "라브라도": {
        "name": "라브라도 리트리버",
        "description": "충실하고 온화한 대형 가정견",
        "face_features": {
            "face_width": "wide",
            "eye_shape": "round",
            "nose_size": "large",
            "mouth_width": "wide",
            "face_length": "medium"
        },
        "personality": ["충실함", "온화함", "사교적"],
        "image": "Labrador_Retriever.png"
    }
}

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
    return DOG_BREEDS.get(breed_name, None)

def get_all_breeds():
    return list(DOG_BREEDS.keys())
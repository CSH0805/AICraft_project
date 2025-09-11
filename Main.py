from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
import math
import os

app = FastAPI(title="🐶 강아지 닮은 얼굴 찾기 API", version="2.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙
if os.path.exists("dog_image"):
    app.mount("/static", StaticFiles(directory="dog_image"), name="static")

# =========================
# 강아지 데이터베이스 (내장)
# =========================
DOG_BREEDS = {
    "골든 리트리버": {
        "name": "골든 리트리버",
        "description": "온순하고 친근한 성격의 대형견",
        "face_features": {"face_width": "wide", "eye_shape": "round", "nose_size": "medium", "mouth_width": "wide", "face_length": "medium"},
        "personality": ["친근함", "온순함", "활발함"],
        "image": "/static/golden_retriever.png"
    },
    "시바견": {
        "name": "시바견", 
        "description": "도도하고 독립적인 성격의 일본 견종",
        "face_features": {"face_width": "narrow", "eye_shape": "narrow", "nose_size": "small", "mouth_width": "small", "face_length": "long"},
        "personality": ["도도함", "독립적", "영리함"],
        "image": "/static/Shiba_Inu.png"
    },
    "푸들": {
        "name": "푸들",
        "description": "영리하고 우아한 성격의 곱슬모 견종", 
        "face_features": {"face_width": "medium", "eye_shape": "oval", "nose_size": "small", "mouth_width": "small", "face_length": "long"},
        "personality": ["영리함", "우아함", "활발함"],
        "image": "/static/poodle.png"
    },
    "불독": {
        "name": "불독",
        "description": "묵직하고 차분한 성격의 단두종",
        "face_features": {"face_width": "very_wide", "eye_shape": "round", "nose_size": "large", "mouth_width": "wide", "face_length": "short"},
        "personality": ["차분함", "묵직함", "충실함"],
        "image": "/static/bulldog.png"
    },
    "비글": {
        "name": "비글",
        "description": "호기심 많고 활발한 중형 사냥견",
        "face_features": {"face_width": "medium", "eye_shape": "round", "nose_size": "medium", "mouth_width": "medium", "face_length": "medium"},
        "personality": ["호기심", "활발함", "사교적"],
        "image": "/static/beagle.png"
    },
    "치와와": {
        "name": "치와와",
        "description": "작지만 용감한 초소형 견종",
        "face_features": {"face_width": "narrow", "eye_shape": "large", "nose_size": "very_small", "mouth_width": "small", "face_length": "short"},
        "personality": ["용감함", "경계심", "애교"],
        "image": "/static/chihuahua.png"
    },
    "허스키": {
        "name": "시베리안 허스키",
        "description": "늑대 같은 외모의 활동적인 견종",
        "face_features": {"face_width": "medium", "eye_shape": "narrow", "nose_size": "medium", "mouth_width": "medium", "face_length": "long"},
        "personality": ["활동적", "독립적", "친근함"],
        "image": "/static/Siberian_Husky.png"
    },
    "라브라도": {
        "name": "라브라도 리트리버",
        "description": "충실하고 온화한 대형 가정견",
        "face_features": {"face_width": "wide", "eye_shape": "round", "nose_size": "large", "mouth_width": "wide", "face_length": "medium"},
        "personality": ["충실함", "온화함", "사교적"],
        "image": "/static/Labrador_Retriever.png"
    }
}

FEATURE_SCORES = {
    "face_width": {"very_wide": 5, "wide": 4, "medium": 3, "narrow": 2, "very_narrow": 1},
    "eye_shape": {"large": 5, "round": 4, "oval": 3, "narrow": 2, "very_narrow": 1},
    "nose_size": {"very_large": 5, "large": 4, "medium": 3, "small": 2, "very_small": 1},
    "mouth_width": {"very_wide": 5, "wide": 4, "medium": 3, "small": 2, "very_small": 1},
    "face_length": {"very_long": 5, "long": 4, "medium": 3, "short": 2, "very_short": 1}
}

# =========================
# MediaPipe 초기화
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# =========================
# 얼굴 분석 함수들
# =========================
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def analyze_face_features(landmarks):
    points = [(lm.x, lm.y) for lm in landmarks]
    features = {}
    
    try:
        # 얼굴 너비
        left_cheek = points[234]
        right_cheek = points[454] 
        face_width = calculate_distance(left_cheek, right_cheek)
        
        if face_width > 0.25:
            features["face_width"] = "very_wide"
        elif face_width > 0.22:
            features["face_width"] = "wide"
        elif face_width > 0.19:
            features["face_width"] = "medium"
        elif face_width > 0.16:
            features["face_width"] = "narrow"
        else:
            features["face_width"] = "very_narrow"
            
        # 눈 모양
        left_eye_left = points[33]
        left_eye_right = points[133]
        left_eye_top = points[159]
        left_eye_bottom = points[145]
        
        eye_width = calculate_distance(left_eye_left, left_eye_right)
        eye_height = calculate_distance(left_eye_top, left_eye_bottom)
        eye_area = eye_width * eye_height
        ratio = eye_width / eye_height if eye_height > 0 else 3
        
        if eye_area > 0.003:
            features["eye_shape"] = "large"
        elif ratio > 3.5:
            features["eye_shape"] = "narrow"
        elif ratio < 2.5:
            features["eye_shape"] = "round"
        else:
            features["eye_shape"] = "oval"
            
        # 코 크기
        nose_left = points[220]
        nose_right = points[440]
        nose_top = points[6]
        nose_bottom = points[2]
        
        nose_width = calculate_distance(nose_left, nose_right)
        nose_height = calculate_distance(nose_top, nose_bottom)
        nose_area = nose_width * nose_height
        
        if nose_area > 0.006:
            features["nose_size"] = "very_large"
        elif nose_area > 0.004:
            features["nose_size"] = "large"
        elif nose_area > 0.002:
            features["nose_size"] = "medium"
        elif nose_area > 0.001:
            features["nose_size"] = "small"
        else:
            features["nose_size"] = "very_small"
            
        # 입 크기
        mouth_left = points[61]
        mouth_right = points[291]
        mouth_width = calculate_distance(mouth_left, mouth_right)
        
        if mouth_width > 0.08:
            features["mouth_width"] = "very_wide"
        elif mouth_width > 0.065:
            features["mouth_width"] = "wide"
        elif mouth_width > 0.05:
            features["mouth_width"] = "medium"
        elif mouth_width > 0.035:
            features["mouth_width"] = "small"
        else:
            features["mouth_width"] = "very_small"
            
        # 얼굴 길이
        forehead = points[10]
        chin = points[175]
        face_length = calculate_distance(forehead, chin)
        
        if face_length > 0.35:
            features["face_length"] = "very_long"
        elif face_length > 0.3:
            features["face_length"] = "long"
        elif face_length > 0.25:
            features["face_length"] = "medium"
        elif face_length > 0.2:
            features["face_length"] = "short"
        else:
            features["face_length"] = "very_short"
            
    except Exception as e:
        print(f"Feature analysis error: {e}")
        features = {"face_width": "medium", "eye_shape": "round", "nose_size": "medium", "mouth_width": "medium", "face_length": "medium"}
    
    return features

def calculate_similarity(human_features, dog_features):
    weights = {"face_width": 0.25, "eye_shape": 0.25, "nose_size": 0.2, "mouth_width": 0.15, "face_length": 0.15}
    total_score = 0
    max_possible_score = 0
    
    for feature, weight in weights.items():
        if feature in human_features and feature in dog_features:
            human_score = FEATURE_SCORES.get(feature, {}).get(human_features[feature], 3)
            dog_score = FEATURE_SCORES.get(feature, {}).get(dog_features[feature], 3)
            diff = abs(human_score - dog_score)
            similarity = max(0, 5 - diff)
            total_score += similarity * weight
            max_possible_score += 5 * weight
    
    return (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0

def find_best_matches(human_features, top_n=3):
    matches = []
    for breed_name, breed_info in DOG_BREEDS.items():
        dog_features = breed_info["face_features"]
        similarity = calculate_similarity(human_features, dog_features)
        
        matching_features = []
        feature_names = {"face_width": "얼굴 너비", "eye_shape": "눈 모양", "nose_size": "코 크기", "mouth_width": "입 크기", "face_length": "얼굴 길이"}
        for feature, human_value in human_features.items():
            if feature in dog_features and human_value == dog_features[feature]:
                matching_features.append(feature_names.get(feature, feature))
        
        matches.append({
            "breed": breed_name,
            "similarity": similarity,
            "description": breed_info["description"],
            "personality": breed_info["personality"],
            "image": breed_info["image"],
            "matching_features": matching_features
        })
    
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches[:top_n]

def get_face_analysis(features):
    width = features.get("face_width", "medium")
    length = features.get("face_length", "medium")
    
    if width in ["wide", "very_wide"] and length == "short":
        face_type = "둥근형"
    elif width in ["narrow", "very_narrow"] and length in ["long", "very_long"]:
        face_type = "긴 얼굴형"
    elif width == "medium" and length == "medium":
        face_type = "표준형"
    elif width in ["wide", "very_wide"]:
        face_type = "넓은 얼굴형"
    else:
        face_type = "독특한 얼굴형"
    
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
    
    recommendations = []
    if width in ["wide", "very_wide"]:
        recommendations.append("넓은 얼굴: 골든 리트리버, 불독, 라브라도와 잘 맞습니다")
    elif width in ["narrow", "very_narrow"]:
        recommendations.append("좁은 얼굴: 시바견, 치와와와 유사한 특징입니다")
    
    return {
        "face_type": face_type,
        "dominant_features": dominant,
        "recommendations": recommendations
    }

# =========================
# 웹페이지
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🐶 강아지 닮은꼴 찾기</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Arial', sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh; padding: 20px;
            }
            .container { 
                max-width: 800px; margin: 0 auto; background: white; 
                border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.2); overflow: hidden;
            }
            .header {
                background: linear-gradient(45deg, #ff6b6b, #feca57);
                padding: 40px; text-align: center; color: white;
            }
            .header h1 { font-size: 2.5rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
            .content { padding: 40px; }
            .upload-section {
                border: 3px dashed #ddd; border-radius: 15px; padding: 40px;
                text-align: center; margin: 30px 0; transition: all 0.3s ease; cursor: pointer;
            }
            .upload-section:hover { border-color: #667eea; background-color: #f8f9ff; }
            .upload-section.dragover { border-color: #667eea; background-color: #f0f4ff; }
            #file-input { display: none; }
            .upload-btn {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white; padding: 15px 30px; border: none; border-radius: 25px;
                font-size: 1.1rem; cursor: pointer; transition: transform 0.3s ease;
            }
            .upload-btn:hover { transform: translateY(-2px); }
            .preview-image { max-width: 200px; max-height: 200px; border-radius: 10px; margin: 20px 0; }
            .loading { display: none; text-align: center; padding: 20px; }
            .spinner {
                border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%;
                width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto 20px;
            }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .result-section { display: none; margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 15px; }
            .match-card {
                background: white; padding: 20px; border-radius: 15px; margin: 15px 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .dog-info { display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
            .dog-image-container {
                flex-shrink: 0; width: 120px; height: 120px; border-radius: 15px; overflow: hidden;
                background: #f8f9fa; display: flex; align-items: center; justify-content: center;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            .dog-image { width: 100%; height: 100%; object-fit: cover; transition: transform 0.3s ease; }
            .dog-image:hover { transform: scale(1.1); }
            .dog-details { flex: 1; min-width: 250px; }
            .breed-name { font-size: 1.4rem; font-weight: bold; margin-bottom: 8px; color: #333; }
            .breed-description { color: #666; margin-bottom: 12px; line-height: 1.4; }
            .similarity-bar { width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; margin: 10px 0; }
            .similarity-fill { height: 100%; background: linear-gradient(45deg, #ff6b6b, #feca57); border-radius: 10px; transition: width 1s ease; }
            .personality-tags { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }
            .personality-tag {
                background: linear-gradient(45deg, #667eea, #764ba2); color: white;
                padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 500;
            }
            .matching-features {
                background: #e8f5e8; padding: 8px 12px; border-radius: 8px; margin-top: 10px;
                border-left: 4px solid #4caf50;
            }
            .matching-features h5 { margin: 0 0 5px 0; color: #2e7d32; font-size: 0.9rem; }
            .matching-features p { margin: 0; color: #388e3c; font-size: 0.85rem; }
            .api-links { display: flex; gap: 15px; justify-content: center; margin-top: 30px; }
            .api-btn {
                padding: 12px 25px; background: #667eea; color: white; text-decoration: none;
                border-radius: 25px; transition: all 0.3s ease;
            }
            .api-btn:hover { background: #764ba2; transform: translateY(-2px); }
            .error-message {
                background: #ffebee; color: #c62828; padding: 15px; border-radius: 8px;
                margin: 20px 0; border-left: 4px solid #f44336;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🐶 강아지 닮은꼴 찾기</h1>
                <p>당신의 얼굴과 가장 닮은 강아지 품종을 찾아보세요!</p>
            </div>
            
            <div class="content">
                <div class="upload-section" onclick="document.getElementById('file-input').click()">
                    <div>🖼️</div>
                    <h3>사진을 업로드하세요</h3>
                    <p>얼굴이 잘 보이는 사진을 선택해주세요 (JPG, PNG, WEBP 형식)</p>
                    <br>
                    <button class="upload-btn" type="button">파일 선택</button>
                    <input type="file" id="file-input" accept="image/jpeg,image/jpg,image/png,image/webp">
                    <img id="preview" class="preview-image" style="display: none;">
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>얼굴을 분석하고 있습니다...</p>
                </div>
                
                <div id="error-container"></div>
                
                <div class="result-section" id="results">
                    <h3>🎯 매칭 결과</h3>
                    <div id="match-results"></div>
                </div>
                
                <div class="api-links">
                    <a href="/docs" class="api-btn">📖 API 문서</a>
                    <a href="/breeds" class="api-btn">🐕 품종 목록</a>
                </div>
            </div>
        </div>

        <script>
            const fileInput = document.getElementById('file-input');
            const preview = document.getElementById('preview');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            const matchResults = document.getElementById('match-results');
            const uploadSection = document.querySelector('.upload-section');
            const errorContainer = document.getElementById('error-container');

            // 드래그 앤 드롭 이벤트
            uploadSection.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadSection.classList.add('dragover');
            });
            
            uploadSection.addEventListener('dragleave', (e) => {
                e.preventDefault();
                uploadSection.classList.remove('dragover');
            });
            
            uploadSection.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadSection.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFile(files[0]);
                }
            });

            // 파일 선택 이벤트
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleFile(e.target.files[0]);
                }
            });

            function showError(message) {
                errorContainer.innerHTML = `<div class="error-message">${message}</div>`;
                loading.style.display = 'none';
                results.style.display = 'none';
            }

            function clearError() {
                errorContainer.innerHTML = '';
            }

            function handleFile(file) {
                clearError();
                
                // 파일 유효성 검사
                const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
                if (!validTypes.includes(file.type)) {
                    showError('지원되는 이미지 형식: JPG, PNG, WEBP');
                    return;
                }
                
                const maxSize = 10 * 1024 * 1024; // 10MB
                if (file.size > maxSize) {
                    showError('파일 크기는 10MB 이하여야 합니다.');
                    return;
                }

                // 미리보기 표시
                const reader = new FileReader();
                reader.onload = (e) => {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
                
                // 분석 시작
                analyzePhoto(file);
            }

            async function analyzePhoto(file) {
                loading.style.display = 'block';
                results.style.display = 'none';

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch("/analyze-face", {
                        method: "POST",
                        body: formData
                    });

                    const data = await response.json();
                    loading.style.display = 'none';

                    if (!response.ok || !data.success) {
                        showError(data.error || '분석 중 오류가 발생했습니다.');
                        return;
                    }

                    displayResults(data);

                } catch (error) {
                    console.error("분석 오류:", error);
                    loading.style.display = 'none';
                    showError('서버 연결 오류가 발생했습니다. 다시 시도해주세요.');
                }
            }

            function displayResults(data) {
                matchResults.innerHTML = "";

                // 얼굴 분석 정보 표시
                if (data.face_analysis) {
                    const analysisCard = document.createElement("div");
                    analysisCard.className = "match-card";
                    
                    let analysisHtml = '<h4>📋 당신의 얼굴 분석</h4><div class="dog-details">';
                    analysisHtml += `<p><strong>얼굴형:</strong> ${data.face_analysis.face_type}</p>`;
                    
                    if (data.face_analysis.dominant_features && data.face_analysis.dominant_features.length > 0) {
                        analysisHtml += `<p><strong>주요 특징:</strong> ${data.face_analysis.dominant_features.join(', ')}</p>`;
                    }
                    
                    analysisHtml += '</div>';
                    analysisCard.innerHTML = analysisHtml;
                    matchResults.appendChild(analysisCard);
                }

                // 매칭 결과 표시
                data.matches.forEach((match, index) => {
                    const rank = index === 0 ? '🥇 1위' : index === 1 ? '🥈 2위' : '🥉 3위';
                    
                    const card = document.createElement("div");
                    card.className = "match-card";

                    card.innerHTML = `
                        <div class="dog-info">
                            <div class="dog-image-container">
                                <img src="${match.image}" alt="${match.breed}" class="dog-image" 
                                     onerror="this.style.display='none'; this.parentElement.innerHTML='🐕';">
                            </div>
                            <div class="dog-details">
                                <div class="breed-name">${rank} ${match.breed}</div>
                                <div class="breed-description">${match.description}</div>
                                <div class="similarity-bar">
                                    <div class="similarity-fill" style="width: ${match.similarity.toFixed(1)}%"></div>
                                </div>
                                <p><strong>유사도:</strong> ${match.similarity.toFixed(1)}%</p>
                                <div class="personality-tags">
                                    ${match.personality.map(p => `<span class="personality-tag">${p}</span>`).join('')}
                                </div>
                                ${match.matching_features.length > 0 ? `
                                <div class="matching-features">
                                    <h5>✨ 일치하는 특징</h5>
                                    <p>${match.matching_features.join(", ")}</p>
                                </div>` : ""}
                            </div>
                        </div>
                    `;

                    matchResults.appendChild(card);
                });

                results.style.display = 'block';
                results.scrollIntoView({ behavior: 'smooth' });
            }
        </script>
    </body>
    </html>
    """

# =========================
# API 엔드포인트
# =========================
@app.post("/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    """얼굴 분석 및 강아지 매칭 API"""
    try:
        # 파일 유효성 검사
        if not file.content_type or not file.content_type.startswith('image/'):
            return JSONResponse(
                content={"success": False, "error": "이미지 파일만 업로드 가능합니다."}, 
                status_code=400
            )
        
        # 파일 크기 제한 (10MB)
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            return JSONResponse(
                content={"success": False, "error": "파일 크기는 10MB 이하여야 합니다."}, 
                status_code=400
            )
        
        # 이미지 디코딩
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(
                content={"success": False, "error": "이미지를 읽을 수 없습니다. 다른 이미지를 시도해보세요."}, 
                status_code=400
            )

        # RGB 변환 및 얼굴 검출
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return JSONResponse(
                content={"success": False, "error": "얼굴을 찾을 수 없습니다. 얼굴이 잘 보이는 사진을 사용해주세요."}, 
                status_code=400
            )

        # 얼굴 특징 분석
        landmarks = results.multi_face_landmarks[0].landmark
        human_features = analyze_face_features(landmarks)
        matches = find_best_matches(human_features, top_n=3)
        face_analysis = get_face_analysis(human_features)
        
        return {
            "success": True,
            "filename": file.filename,
            "human_features": human_features,
            "face_analysis": face_analysis,
            "matches": matches
        }
        
    except Exception as e:
        print(f"Error in analyze_face: {str(e)}")
        return JSONResponse(
            content={"success": False, "error": f"분석 중 오류가 발생했습니다: {str(e)}"}, 
            status_code=500
        )

@app.post("/find_similar_dog")
async def find_similar_dog(file: UploadFile = File(...)):
    """기존 API 호환성을 위한 엔드포인트"""
    return await analyze_face(file)

@app.get("/breeds")
def get_breeds():
    """강아지 품종 목록 조회"""
    return {
        "breeds": list(DOG_BREEDS.keys()),
        "total_breeds": len(DOG_BREEDS),
        "breed_details": DOG_BREEDS
    }

@app.get("/health")
def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "message": "강아지 닮은꼴 찾기 API가 정상 작동 중입니다!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)



# 실행 방법:
# uvicorn main:app --reload
# 또는 python main.py
# http://127.0.0.1:8000/ 에서 웹 인터페이스 사용
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import mediapipe as mp

# 우리가 만든 모듈들 import
from face_analyzer import FaceAnalyzer
from dog_matcher import DogMatcher
from dog_database import DOG_BREEDS, get_all_breeds

app = FastAPI(title="🐶 강아지 닮은 얼굴 찾기 API",
              description="사람 얼굴을 분석해서 가장 닮은 강아지 품종을 찾아주는 API",
              version="2.0")

# =========================
# MediaPipe & 분석기 초기화
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

face_analyzer = FaceAnalyzer()
dog_matcher = DogMatcher()

# =========================
# 루트 경로: 메인 웹페이지
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
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
                min-height: 100vh;
                padding: 20px;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 20px; 
                box-shadow: 0 20px 60px rgba(0,0,0,0.2);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(45deg, #ff6b6b, #feca57);
                padding: 40px;
                text-align: center;
                color: white;
            }
            .header h1 { 
                font-size: 2.5rem; 
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .content { padding: 40px; }
            .upload-section {
                border: 3px dashed #ddd;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                margin: 30px 0;
                transition: all 0.3s ease;
                cursor: pointer;
            }
            .upload-section:hover {
                border-color: #667eea;
                background-color: #f8f9ff;
            }
            .upload-section.dragover {
                border-color: #667eea;
                background-color: #f0f4ff;
            }
            #file-input {
                display: none;
            }
            .upload-btn {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 25px;
                font-size: 1.1rem;
                cursor: pointer;
                transition: transform 0.3s ease;
            }
            .upload-btn:hover {
                transform: translateY(-2px);
            }
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            .feature-card {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                border: 2px solid transparent;
                transition: all 0.3s ease;
            }
            .feature-card:hover {
                border-color: #667eea;
                transform: translateY(-5px);
            }
            .api-links {
                display: flex;
                gap: 15px;
                justify-content: center;
                margin-top: 30px;
            }
            .api-btn {
                padding: 12px 25px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 25px;
                transition: all 0.3s ease;
            }
            .api-btn:hover {
                background: #764ba2;
                transform: translateY(-2px);
            }
            .result-section {
                display: none;
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 15px;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .match-card {
                background: white;
                padding: 20px;
                border-radius: 15px;
                margin: 15px 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .similarity-bar {
                width: 100%;
                height: 20px;
                background: #e9ecef;
                border-radius: 10px;
                overflow: hidden;
                margin: 10px 0;
            }
            .similarity-fill {
                height: 100%;
                background: linear-gradient(45deg, #ff6b6b, #feca57);
                border-radius: 10px;
                transition: width 1s ease;
            }
            .dog-info {
                display: flex;
                gap: 20px;
                align-items: center;
            }
            .dog-emoji {
                font-size: 4rem;
                margin-right: 20px;
            }
            .preview-image {
                max-width: 200px;
                max-height: 200px;
                border-radius: 10px;
                margin: 20px 0;
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
                    <p>얼굴이 잘 보이는 사진을 선택해주세요</p>
                    <br>
                    <button class="upload-btn" type="button">파일 선택</button>
                    <input type="file" id="file-input" accept="image/*">
                    <img id="preview" class="preview-image" style="display: none;">
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>얼굴을 분석하고 있습니다...</p>
                </div>
                
                <div class="result-section" id="results">
                    <h3>🎯 매칭 결과</h3>
                    <div id="match-results"></div>
                </div>
                
                <div class="features">
                    <div class="feature-card">
                        <h4>🔍 얼굴 분석</h4>
                        <p>MediaPipe를 사용해 468개의 얼굴 랜드마크를 추출합니다</p>
                    </div>
                    <div class="feature-card">
                        <h4>🐕 품종 매칭</h4>
                        <p>8가지 강아지 품종과 특징을 비교분석합니다</p>
                    </div>
                    <div class="feature-card">
                        <h4>📊 유사도 계산</h4>
                        <p>얼굴형, 눈, 코, 입 특징을 종합해 매칭률을 계산합니다</p>
                    </div>
                </div>
                
                <div class="api-links">
                    <a href="/docs" class="api-btn">📖 API 문서</a>
                    <a href="/breeds" class="api-btn">🐶 강아지 품종</a>
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

            // 드래그 앤 드롭 처리
            uploadSection.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadSection.classList.add('dragover');
            });

            uploadSection.addEventListener('dragleave', () => {
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

            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleFile(e.target.files[0]);
                }
            });

            function handleFile(file) {
                // 이미지 미리보기
                const reader = new FileReader();
                reader.onload = (e) => {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);

                // API 호출
                analyzePhoto(file);
            }

            async function analyzePhoto(file) {
                loading.style.display = 'block';
                results.style.display = 'none';

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch('/find_similar_dog', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    
                    if (data.error) {
                        alert('오류: ' + data.error);
                        return;
                    }

                    displayResults(data);
                } catch (error) {
                    alert('분석 중 오류가 발생했습니다: ' + error.message);
                } finally {
                    loading.style.display = 'none';
                }
            }

            function displayResults(data) {
                let html = '';
                
                // 분석된 특징 표시
                html += '<div class="match-card">';
                html += '<h4>📋 당신의 얼굴 특징</h4>';
                html += '<div class="dog-info">';
                html += '<div>';
                html += '<p><strong>얼굴형:</strong> ' + data.face_analysis.face_type + '</p>';
                if (data.face_analysis.dominant_features.length > 0) {
                    html += '<p><strong>주요 특징:</strong> ' + data.face_analysis.dominant_features.join(', ') + '</p>';
                }
                html += '</div>';
                html += '</div>';
                html += '</div>';

                // 매칭 결과 표시
                data.matches.forEach((match, index) => {
                    const emoji = index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉';
                    html += '<div class="match-card">';
                    html += '<div class="dog-info">';
                    html += '<div class="dog-emoji">' + getDogEmoji(match.breed) + '</div>';
                    html += '<div style="flex: 1;">';
                    html += '<h4>' + emoji + ' ' + match.breed + '</h4>';
                    html += '<p>' + match.description + '</p>';
                    html += '<div class="similarity-bar">';
                    html += '<div class="similarity-fill" style="width: ' + match.similarity.toFixed(1) + '%"></div>';
                    html += '</div>';
                    html += '<p><strong>유사도:</strong> ' + match.similarity.toFixed(1) + '%</p>';
                    html += '<p><strong>성격:</strong> ' + match.personality.join(', ') + '</p>';
                    if (match.matching_features.length > 0) {
                        html += '<p><strong>일치하는 특징:</strong> ' + match.matching_features.join(', ') + '</p>';
                    }
                    html += '</div>';
                    html += '</div>';
                    html += '</div>';
                });

                matchResults.innerHTML = html;
                results.style.display = 'block';
                results.scrollIntoView({ behavior: 'smooth' });
            }

            function getDogEmoji(breed) {
                const emojis = {
                    '골든 리트리버': '🐕',
                    '시바견': '🦊',
                    '푸들': '🐩',
                    '불독': '🐶',
                    '비글': '🐕',
                    '치와와': '🐕‍🦺',
                    '시베리안 허스키': '🐺',
                    '라브라도 리트리버': '🦮'
                };
                return emojis[breed] || '🐶';
            }
        </script>
    </body>
    </html>
    """
    return html_content

# =========================
# 강아지 품종 정보 API
# =========================
@app.get("/breeds")
def get_breeds():
    return {"breeds": list(DOG_BREEDS.keys()), "total": len(DOG_BREEDS)}

@app.get("/breeds/{breed_name}")
def get_breed_info(breed_name: str):
    breed_info = DOG_BREEDS.get(breed_name)
    if not breed_info:
        return JSONResponse(content={"error": "품종을 찾을 수 없습니다."}, status_code=404)
    return breed_info

# =========================
# 얼굴 랜드마크 추출 API (기존)
# =========================
@app.post("/extract_landmarks")
async def extract_landmarks(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                content={"error": "이미지 파일만 업로드 가능합니다."}, 
                status_code=400
            )
        
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(
                content={"error": "이미지를 읽을 수 없습니다."}, 
                status_code=400
            )

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return JSONResponse(
                content={"error": "얼굴을 찾을 수 없습니다."}, 
                status_code=400
            )

        landmarks = results.multi_face_landmarks[0].landmark
        landmark_points = [
            {"x": lm.x, "y": lm.y, "z": lm.z}
            for lm in landmarks
        ]

        return {
            "success": True,
            "filename": file.filename,
            "total_landmarks": len(landmark_points),
            "landmarks": landmark_points
        }
        
    except Exception as e:
        return JSONResponse(
            content={"error": f"처리 중 오류가 발생했습니다: {str(e)}"}, 
            status_code=500
        )

# =========================
# 🎯 메인 기능: 닮은 강아지 찾기 API
# =========================
@app.post("/find_similar_dog")
async def find_similar_dog(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                content={"error": "이미지 파일만 업로드 가능합니다."}, 
                status_code=400
            )
        
        # 1. 이미지에서 얼굴 랜드마크 추출
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(
                content={"error": "이미지를 읽을 수 없습니다."}, 
                status_code=400
            )

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return JSONResponse(
                content={"error": "얼굴을 찾을 수 없습니다. 얼굴이 잘 보이는 사진을 업로드해주세요."}, 
                status_code=400
            )

        landmarks = results.multi_face_landmarks[0].landmark
        
        # 2. 얼굴 특징 분석
        human_features = face_analyzer.analyze_face_features(landmarks)
        if not human_features:
            return JSONResponse(
                content={"error": "얼굴 특징을 분석할 수 없습니다."}, 
                status_code=400
            )
        
        # 3. 강아지 매칭
        matches = dog_matcher.find_best_matches(human_features, top_n=3)
        face_analysis = dog_matcher.get_detailed_analysis(human_features)
        
        return {
            "success": True,
            "filename": file.filename,
            "human_features": human_features,
            "face_analysis": face_analysis,
            "matches": matches,
            "total_breeds_analyzed": len(DOG_BREEDS)
        }
        
    except Exception as e:
        return JSONResponse(
            content={"error": f"분석 중 오류가 발생했습니다: {str(e)}"}, 
            status_code=500
        )

# =========================
# JSON 응답 API (기존 호환성)
# =========================
@app.get("/api")
def home_json():
    return {
        "message": "🐶 강아지 닮은꼴 찾기 API",
        "version": "2.0",
        "endpoints": {
            "find_similar_dog": "POST /find_similar_dog - 메인 기능",
            "extract_landmarks": "POST /extract_landmarks - 랜드마크 추출",
            "breeds": "GET /breeds - 강아지 품종 목록",
            "breed_info": "GET /breeds/{breed_name} - 특정 품종 정보"
        }
    }

#uvicorn Main:app --reload
# http://127.0.0.1:8000/docs
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Allow CORS for local testing (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("models/imageclassifier.h5")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"UTF-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
        <title>Happy or Not Classifier</title>
        <link href=\"https://fonts.googleapis.com/css2?family=Montserrat:wght@700&family=Roboto:wght@400;500&display=swap\" rel=\"stylesheet\">
        <style>
            body {
                background: linear-gradient(135deg, #f8ffae 0%, #43c6ac 100%);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                font-family: 'Roboto', Arial, sans-serif;
                margin: 0;
            }
            .container {
                background: rgba(255,255,255,0.95);
                border-radius: 20px;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
                padding: 40px 32px 32px 32px;
                max-width: 400px;
                width: 100%;
                text-align: center;
            }
            h2 {
                font-family: 'Montserrat', sans-serif;
                font-size: 2rem;
                color: #2d3a4b;
                margin-bottom: 24px;
            }
            .upload-btn-wrapper {
                position: relative;
                overflow: hidden;
                display: inline-block;
                margin-bottom: 24px;
            }
            .btn {
                border: none;
                outline: none;
                color: white;
                background: linear-gradient(90deg, #43c6ac 0%, #191654 100%);
                padding: 14px 32px;
                border-radius: 30px;
                font-size: 1.1rem;
                font-family: 'Montserrat', sans-serif;
                font-weight: 700;
                cursor: pointer;
                transition: background 0.3s, transform 0.2s;
                box-shadow: 0 4px 16px rgba(67,198,172,0.15);
            }
            .btn:hover {
                background: linear-gradient(90deg, #191654 0%, #43c6ac 100%);
                transform: translateY(-2px) scale(1.03);
            }
            input[type=file] {
                font-size: 1rem;
                position: absolute;
                left: 0;
                top: 0;
                opacity: 0;
                width: 100%;
                height: 100%;
                cursor: pointer;
            }
            #result {
                margin-top: 28px;
                font-size: 1.3rem;
                font-weight: 500;
                color: #191654;
                min-height: 32px;
                transition: color 0.3s;
            }
            .preview-img {
                margin-top: 18px;
                max-width: 180px;
                max-height: 180px;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(67,198,172,0.12);
                display: none;
            }
        </style>
    </head>
    <body>
        <div class=\"container\">
            <h2>Happy or Not Classifier</h2>
            <form id=\"upload-form\" enctype=\"multipart/form-data\" method=\"post\">
                <div class=\"upload-btn-wrapper\">
                    <button class=\"btn\" type=\"button\" onclick=\"document.getElementById('file-input').click();\">Choose Image</button>
                    <input id=\"file-input\" name=\"file\" type=\"file\" accept=\"image/*\" required onchange=\"showPreview(event)\">
                </div>
                <br>
                <img id=\"preview\" class=\"preview-img\"/>
                <br>
                <button class=\"btn\" type=\"submit\">Classify</button>
            </form>
            <div id=\"result\"></div>
        </div>
        <script>
            function showPreview(event) {
                const preview = document.getElementById('preview');
                const file = event.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                    }
                    reader.readAsDataURL(file);
                } else {
                    preview.src = '';
                    preview.style.display = 'none';
                }
            }
            document.getElementById('upload-form').onsubmit = async function(e) {
                e.preventDefault();
                const form = e.target;
                const fileInput = document.getElementById('file-input');
                if (!fileInput.files.length) return;
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                document.getElementById('result').textContent = 'Classifying...';
                try {
                    const response = await fetch('/predict/', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    let emoji = data.prediction === 'Happy' ? '😊' : '😢';
                    document.getElementById('result').innerHTML = `<b>Result:</b> ${data.prediction} ${emoji}<br><span style='font-size:0.95em;color:#555;'>Confidence: ${(data.probability*100).toFixed(1)}%</span>`;
                } catch (err) {
                    document.getElementById('result').textContent = 'Error: Could not classify image.';
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = preprocess_image(image_bytes)
    pred = model.predict(img)[0][0]
    if pred < 0.5:
     result = "Happy"
     confidence = 1 - pred
    else:
     result = "Sad"
     confidence = pred

    return {"prediction": result, "probability": float(confidence)}

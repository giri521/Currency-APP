import os
import json
import base64
import time
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
import io
import requests

# --- Load Environment Variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
CORS(app)

# --- Gemini API Configuration ---
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
GEMINI_API_URL = f"{GEMINI_API_BASE_URL}{MODEL_NAME}:generateContent"

SYSTEM_PROMPT = """
You are a highly accurate currency note identifier. You are given an image of an Indian banknote (₹10, ₹20, ₹50, ₹100, ₹200, ₹500). The note can be front or back.

Your task is to:
1. Identify the side (front/back) and denomination (10, 20, 50, 100, 200, 500).
2. Set 'full_validation' to true only if the side and denomination are clearly verifiable.
3. Generate 'speech_text' stating the denomination.

Output ONLY a strictly valid JSON object:
{
  "side": "front" | "back",
  "denomination": 10 | 20 | 50 | 100 | 200 | 500 | "null",
  "full_validation": true | false,
  "speech_text": "It is a <denomination> Rupees note." | "Note not clear, please show the note fully."
}
If detection fails, set 'denomination' to "null".
"""

# --- Utility for API Retry ---
def call_api_with_retry(url, headers, json_payload, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=json_payload)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [429, 500, 502, 503, 504] and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    raise requests.exceptions.RequestException("Max retries exceeded.")

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_currency():
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API Key not configured.", "speech_text": "API key missing."}), 500
    if 'image' not in request.files:
        return jsonify({"error": "No image provided.", "speech_text": "No image received."}), 400

    image_file = request.files['image']

    try:
        # Convert image to base64
        img = Image.open(io.BytesIO(image_file.read()))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "Analyze this Indian banknote image and return JSON exactly as requested."},
                        {"inlineData": {"mimeType": "image/jpeg", "data": img_str}}
                    ]
                }
            ],
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]}
        }

        headers = {"Content-Type": "application/json"}
        full_api_url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

        response = call_api_with_retry(full_api_url, headers, payload)

        raw_text = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        json_text = raw_text.replace('```json','').replace('```','').strip()
        result_json = json.loads(json_text)

        # Ensure full_validation is boolean
        result_json['full_validation'] = bool(result_json.get('full_validation', False))
        result_json['denomination'] = str(result_json.get('denomination', "null"))

        return jsonify(result_json), 200

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON returned by Gemini.", "speech_text": "Analysis failed."}), 500
    except Exception as e:
        return jsonify({"error": str(e), "speech_text": "Internal server error."}), 500

# --- Main ---
if __name__ == '__main__':
    print("Starting Flask server for Gemini API...")
    app.run(host='0.0.0.0', port=5000, debug=True)


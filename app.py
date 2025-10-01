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
# Note: Ensure you replace PERPLEXITY_API_KEY with GEMINI_API_KEY in your .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for mobile/web frontend

# --- Gemini API Configuration ---
# Using the model for vision and structured JSON output
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
GEMINI_API_URL = f"{GEMINI_API_BASE_URL}{MODEL_NAME}:generateContent"

# --- System Prompt (Remains focused on JSON structure) ---
SYSTEM_PROMPT = """
You are a highly accurate currency note identifier. You are given an image of an Indian banknote (₹10, ₹20, ₹50, ₹100, ₹200, ₹500). The note can be front or back.

Your task is to:
1. Identify the side (front/back) and denomination (10, 20, 50, 100, 200, 500).
2. Set 'full_validation' to true only if the side and denomination are clearly verifiable (e.g., Gandhi + RBI for front, Monument + Denomination for back).
3. Generate a 'speech_text' output that clearly states the denomination.

Output ONLY a single, strictly valid JSON object. DO NOT include any introductory or explanatory text or surrounding markdown code blocks.
{
  "side": "front" | "back",
  "denomination": 10 | 20 | 50 | 100 | 200 | 500 | "null",
  "full_validation": true | false,
  "speech_text": "It is a <denomination> Rupees note." | "Note not clear, please show the note fully."
}
If detection fails, set 'denomination' to "null" and provide the appropriate 'speech_text'.
"""

# --- Utility for Exponential Backoff (for API resilience) ---
def call_api_with_retry(url, headers, json_payload, max_retries=5):
    """Calls the Gemini API with exponential backoff for rate limit handling."""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=json_payload)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            # Check for retryable status codes (e.g., 429 Too Many Requests, 5xx server errors)
            if e.response.status_code in [429, 500, 502, 503, 504] and attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                print(f"Rate limit or server error ({e.response.status_code}). Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                raise # Re-raise for non-retryable errors or final attempt failure
    raise requests.exceptions.RequestException("Max retries exceeded.")


# --- Routes ---
@app.route('/')
def index():
    # Renders the HTML template located in the 'templates' folder
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_currency():
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API Key not configured in .env.",
                        "speech_text": "Error: API key not configured."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image provided.",
                        "speech_text": "Error: No image received."}), 400

    image_file = request.files['image']

    try:
        # --- Convert image to Base64 ---
        img = Image.open(io.BytesIO(image_file.read()))
        buffer = io.BytesIO()
        # Save as JPEG to ensure 'image/jpeg' mimeType
        img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # --- Prepare Gemini API Payload (Multi-part request with JSON schema) ---
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        # Text part
                        {"text": "Analyze this Indian banknote image and return the JSON exactly as requested in the system prompt."},
                        # Image part (inlineData)
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": img_str
                            }
                        }
                    ]
                }
            ],
            # System Instruction for Model Behavior
            "systemInstruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            },
            # Configuration for structured JSON output
            "config": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "side": {"type": "STRING", "description": "The side of the note: front or back."},
                        "denomination": {"type": ["INTEGER", "STRING"], "description": "The denomination (10, 20, 50, 100, 200, 500) or 'null' if undetectable."},
                        "full_validation": {"type": "BOOLEAN", "description": "True if the note's identity is clearly verifiable."},
                        "speech_text": {"type": "STRING", "description": "A clear, concise spoken summary of the finding."}
                    },
                    "required": ["side", "denomination", "full_validation", "speech_text"]
                }
            }
        }

        # Gemini API typically uses key in URL
        full_api_url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}

        # --- Call Gemini API with Retry Logic ---
        response = call_api_with_retry(full_api_url, headers, payload)

        # --- Parse Gemini Response ---
        raw_text = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        
        # The model is forced to output JSON, but we clean and load it defensively
        try:
            # Clean up potential markdown wrapper (shouldn't be needed with responseMimeType, but safer)
            json_text = raw_text.replace('```json','').replace('```','').strip()
            result_json = json.loads(json_text)
            
            # Convert denomination to string for consistency if model returns int
            if isinstance(result_json.get('denomination'), int):
                result_json['denomination'] = str(result_json['denomination'])
                
            return jsonify(result_json), 200
        except json.JSONDecodeError as e:
            return jsonify({
                "error": f"Gemini returned invalid JSON: {e}",
                "raw_response": raw_text,
                "speech_text": "Analysis failed. Server returned unstructured response."
            }), 500

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_detail = e.response.json().get('error', {}).get('message', 'Unknown API Error')
        
        if status_code == 400:
            speech_err = f"Bad request. Detail: {error_detail[:50]}..."
        elif status_code == 401:
            speech_err = "API key error. Check your Gemini key."
        elif status_code == 429:
            speech_err = "Rate limit exceeded. Too many requests."
        else:
            speech_err = f"Gemini API Error: {status_code}"
            
        return jsonify({"error": str(e), "speech_text": speech_err}), status_code

    except Exception as e:
        return jsonify({"error": f"Server processing error: {e}", "speech_text": "Internal server error."}), 500


# --- Main ---
if __name__ == '__main__':
    print("Starting Flask server for Gemini API...")
    # NOTE: In a production environment, use environment variables for host/port.
    # Running on 0.0.0.0 makes it accessible outside the local machine (useful for testing).
    app.run(host='0.0.0.0', port=5000, debug=True)

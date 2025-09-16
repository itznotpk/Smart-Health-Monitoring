from flask import Flask, render_template_string, request, stream_with_context, Response
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import logging
import re
import pdfplumber
import os
from werkzeug.utils import secure_filename
import ollama
import requests
import json

# ==========================
# Ollama Streaming Integration
# ==========================
def stream_from_ollama(prompt, model="llama3:latest"):
    """
    Generator that streams tokens from Ollama API.
    """
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": True}

    with requests.post(url, json=payload, stream=True) as r:
        for line in r.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    yield data["response"]  # stream each token
                if data.get("done", False):
                    break

# ==========================
# Flask App
# ==========================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set up logging to capture errors
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================
# Load Pre-trained Model and Preprocessor
# ==========================
try:
    model = load_model('diabetes_ann_model.h5')  # Load the trained ANN
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

try:
    preprocessor = joblib.load('preprocessor.joblib')  # Load the preprocessor
    logger.info("Preprocessor loaded successfully")
except Exception as e:
    logger.error(f"Failed to load preprocessor: {str(e)}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ==========================
# Web UI
# ==========================
html_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Universiti Malaya Diabetes Risk Kiosk from Clinical Report</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; background-color: #f8f9fa; }
        .container { margin-top: 30px; max-width: 600px; margin-left: auto; margin-right: auto; }
        .form-group { margin: 15px 0; display: flex; flex-direction: column; align-items: center; }
        .form-group label { font-weight: bold; margin-bottom: 5px; text-align: center; width: 100%; }
        input[type="file"], select { padding: 10px; width: 250px; border: 1px solid #ccc; border-radius: 4px; font-size: 16px; }
        .error-message { color: darkred; font-size: 14px; margin-top: 5px; display: none; }
        .buttons { margin-top: 20px; display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }
        button { padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; font-weight: bold; }
        button[type="submit"] { background-color: #007bff; color: white; }
        button[type="submit"]:hover { background-color: #0056b3; }
        button[type="submit"]:disabled { background-color: #cccccc; cursor: not-allowed; }
        button[type="reset"], button[type="button"] { background-color: #6c757d; color: white; }
        button[type="reset"]:hover, button[type="button"]:hover { background-color: #545b62; }
        .result { font-size: 22px; margin-top: 20px; font-weight: bold; padding: 15px; border-radius: 4px; }
        .explanation { font-size: 16px; margin-top: 10px; padding: 10px; background-color: #e9ecef; border-radius: 4px; }
        .green { color: green; background-color: #d4edda; }
        .red { color: red; background-color: #f8d7da; }
        .orange { color: orange; background-color: #fff3cd; }
        .black { color: black; }
        .error { color: darkred; font-weight: bold; background-color: #f8d7da; padding: 10px; border-radius: 4px; margin: 10px 0; }
        table { margin: 20px auto; border-collapse: collapse; width: 80%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .guideline-box { border: 1px solid #ddd; padding: 10px; margin: 20px auto; width: 80%; border-radius: 4px; background-color: #f8f9fa; }
        .guideline-box h3 { cursor: pointer; margin: 0; padding: 10px; background-color: #e9ecef; border-radius: 4px; }
        .guideline-box h3:hover { background-color: #d8dfe6; }
        .guideline-table { transition: max-height 0.3s ease, opacity 0.3s ease; max-height: 0; opacity: 0; overflow: hidden; }
        .guideline-table.visible { max-height: 500px; opacity: 1; }
        .null-cell { background-color: #cccccc; }
        .results-section { margin-top: 30px; }
        .hidden { display: none !important; }
        #loadingSpinner { margin-top: 20px; font-size: 16px; color: #007bff; }
        .spinner { display: inline-block; border: 4px solid #f3f3f3; border-top: 4px solid #007bff; border-radius: 50%; width: 20px; height: 20px; animation: spin 1s linear infinite; margin-left: 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <h1>AI-Powered Diabetes Risk Kiosk from Clinical Report</h1>
    <div class="container">
        <p><strong>Disclaimer:</strong> This tool is for informational purposes only and uses Malaysian clinical guidelines. Consult a doctor for a proper diagnosis.</p>
        <form method="post" enctype="multipart/form-data" id="analysisForm" onsubmit="showLoading()">
            <div class="form-group">
                <label for="file">Upload Clinical Report (PDF):</label>
                <input type="file" name="file" id="file" accept=".pdf" required aria-label="Upload clinical report PDF">
                <span id="file-error" class="error-message">Please upload a PDF file.</span>
            </div>
            <div class="form-group">
                <label for="heart_disease">Heart Disease (Past/Current):</label>
                <select name="heart_disease" id="heart_disease" required aria-label="Select heart disease status">
                    <option value="">Select</option>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                </select>
                <span id="heart-disease-error" class="error-message">Please select an option.</span>
            </div>
            <div class="form-group">
                <label for="smoking_history">Smoking History:</label>
                <select name="smoking_history" id="smoking_history" required aria-label="Select smoking history">
                    <option value="">Select</option>
                    <option value="No Info">No Info</option>
                    <option value="never">Never</option>
                    <option value="former">Former</option>
                    <option value="current">Current</option>
                    <option value="not current">Not Current</option>
                </select>
                <span id="smoking-history-error" class="error-message">Please select an option.</span>
            </div>
            <div class="buttons">
                <button type="submit" id="submitBtn" disabled>Analyze Report and Predict Risk</button>
                <button type="reset" onclick="resetResults()">Reset Form</button>
            </div>
        </form>
        <div id="loadingSpinner" class="hidden">Processing... <span class="spinner"></span></div>

        {% if result %} 
        <div class="results-section">
            <div id="ollama-output" style="white-space: pre-wrap; border: 1px solid #ccc; padding: 10px; margin-top: 15px;">
                <strong>AI Recommendations will appear here:</strong>
            </div>
        </div>

        <script>
        function startStreaming() {
            const eventSource = new EventSource("/stream_recommendation");
            const outputDiv = document.getElementById("ollama-output");
            outputDiv.innerHTML = "";

            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === "result") {
                    outputDiv.innerHTML += data.text;  // Append streamed text
                } else if (data.type === "end") {
                    eventSource.close();
                } else if (data.type === "error") {
                    outputDiv.innerHTML = "<span style='color:red;'>" + data.text + "</span>";
                    eventSource.close();
                }
            };

            eventSource.onerror = function() {
                outputDiv.innerHTML += "<br><span style='color:red;'>Error receiving AI response.</span>";
                eventSource.close();
            };
        }

        // Trigger only if there is a prediction result
        document.addEventListener("DOMContentLoaded", function() {
            {% if result %}
                startStreaming();
            {% endif %}
        });
        </script>
        {% endif %}

        {% if diagnosis %}
        <div id="diagnosisTableDiv" class="results-section">
            <table>
                <tr><th>Metric</th><th>Value</th><th>Unit</th><th>Status</th></tr>
                {% if 'age' in diagnosis %}
                <tr>
                    <td>Age</td>
                    <td>{{ diagnosis.age.value }}</td>
                    <td>{{ diagnosis.age.unit }}</td>
                    <td class="{{ diagnosis.age.color }}">{{ diagnosis.age.status }}</td>
                </tr>
                {% else %}
                <tr><td>Age</td><td class="null-cell"></td><td class="null-cell"></td><td class="null-cell"></td></tr>
                {% endif %}
                {% if 'sex' in diagnosis %}
                <tr>
                    <td>Gender</td>
                    <td>{{ diagnosis.sex.value }}</td>
                    <td>{{ diagnosis.sex.unit }}</td>
                    <td class="{{ diagnosis.sex.color }}">{{ diagnosis.sex.status }}</td>
                </tr>
                {% else %}
                <tr><td>Gender</td><td class="null-cell"></td><td class="null-cell"></td><td class="null-cell"></td></tr>
                {% endif %}
                {% if 'bmi' in diagnosis %}
                <tr>
                    <td>BMI</td>
                    <td>{{ diagnosis.bmi.value | round(1) }}</td>
                    <td>{{ diagnosis.bmi.unit }}</td>
                    <td class="{{ diagnosis.bmi.color }}">{{ diagnosis.bmi.status }}</td>
                </tr>
                {% else %}
                <tr><td>BMI</td><td class="null-cell"></td><td class="null-cell"></td><td class="null-cell"></td></tr>
                {% endif %}
                {% if 'heart_disease' in diagnosis %}
                <tr>
                    <td>Heart Disease</td>
                    <td>{{ diagnosis.heart_disease.value }}</td>
                    <td>{{ diagnosis.heart_disease.unit }}</td>
                    <td class="{{ diagnosis.heart_disease.color }}">{{ diagnosis.heart_disease.status }}</td>
                </tr>
                {% else %}
                <tr><td>Heart Disease</td><td class="null-cell"></td><td class="null-cell"></td><td class="null-cell"></td></tr>
                {% endif %}
                {% if 'smoking_history' in diagnosis %}
                <tr>
                    <td>Smoking History</td>
                    <td>{{ diagnosis.smoking_history.value }}</td>
                    <td>{{ diagnosis.smoking_history.unit }}</td>
                    <td class="{{ diagnosis.smoking_history.color }}">{{ diagnosis.smoking_history.status }}</td>
                </tr>
                {% else %}
                <tr><td>Smoking History</td><td class="null-cell"></td><td class="null-cell"></td><td class="null-cell"></td></tr>
                {% endif %}
                {% if 'hypertension' in diagnosis %}
                <tr>
                    <td>Hypertension</td>
                    <td>{{ diagnosis.hypertension.value }}</td>
                    <td>{{ diagnosis.hypertension.unit }}</td>
                    <td class="{{ diagnosis.hypertension.color }}">{{ diagnosis.hypertension.status }}</td>
                </tr>
                {% else %}
                <tr><td>Hypertension</td><td class="null-cell"></td><td class="null-cell"></td><td class="null-cell"></td></tr>
                {% endif %}
                {% if 'glucose' in diagnosis %}
                <tr>
                    <td>Blood Glucose ({{ diagnosis.glucose.category }})</td>
                    <td>{{ diagnosis.glucose.value_mg | round(1) }}</td>
                    <td>mg/dL</td>
                    <td rowspan="2" class="{{ diagnosis.glucose.color }}">{{ diagnosis.glucose.status }}</td>
                </tr>
                <tr>
                    <td></td>
                    <td>{{ diagnosis.glucose.value_mmol | round(1) }}</td>
                    <td>mmol/L</td>
                </tr>
                {% endif %}
                {% if 'hba1c' in diagnosis %}
                <tr>
                    <td>HbA1c Level</td>
                    <td>{{ diagnosis.hba1c.value_percent | round(1) }}</td>
                    <td>%</td>
                    <td rowspan="2" class="{{ diagnosis.hba1c.color }}">{{ diagnosis.hba1c.status }}</td>
                </tr>
                <tr>
                    <td></td>
                    <td>{{ diagnosis.hba1c.value_mmol | round(0) }}</td>
                    <td>mmol/mol</td>
                </tr>
                {% endif %}
            </table>
        </div>
        {% endif %}

        <div class="guideline-box">
            <h3 onclick="toggleGuideline(this)">Hypertension Benchmarks (mmHg)</h3>
            <table class="guideline-table">
                <tr><th>Status</th><th>Range</th></tr>
                <tr><td>Normal</td><td>&lt;120/&lt;80</td></tr>
                <tr><td>Elevated/Prehypertension</td><td>120-139/80-89</td></tr>
                <tr><td>Hypertension Stage 1</td><td>140-159/90-99</td></tr>
                <tr><td>Hypertension Stage 2</td><td>&ge;160/&ge;100</td></tr>
            </table>
        </div>
        <div class="guideline-box">
            <h3 onclick="toggleGuideline(this)">BMI Benchmarks (kg/m²)</h3>
            <table class="guideline-table">
                <tr><th>Status</th><th>Range</th></tr>
                <tr><td>Underweight</td><td>&lt;18.5</td></tr>
                <tr><td>Normal range</td><td>18.5-24.9</td></tr>
                <tr><td>Overweight</td><td>&ge;25.0</td></tr>
                <tr><td>Pre-obese</td><td>25.0-29.9</td></tr>
                <tr><td>Obese class I</td><td>30.0-34.9</td></tr>
                <tr><td>Obese class II</td><td>35.0-39.9</td></tr>
                <tr><td>Obese class III</td><td>&ge;40.0</td></tr>
            </table>
        </div>
        <div class="guideline-box">
            <h3 onclick="toggleGuideline(this)">Glucose Benchmarks (mmol/L)</h3>
            <table class="guideline-table">
                <tr><th>Category</th><th>Normal</th><th>Prediabetes</th><th>T2DM Diagnosis</th></tr>
                <tr><td>Fasting</td><td>3.9-6.0</td><td>6.1-6.9</td><td>&ge;7.0</td></tr>
                <tr><td>Random</td><td>3.9-7.7</td><td>7.8-11.0</td><td>&ge;11.1</td></tr>
            </table>
        </div>
        <div class="guideline-box">
            <h3 onclick="toggleGuideline(this)">HbA1c Benchmarks</h3>
            <table class="guideline-table">
                <tr><th>Status</th><th>%</th><th>mmol/mol</th></tr>
                <tr><td>Normal</td><td>&lt;5.7</td><td>&lt;39</td></tr>
                <tr><td>Prediabetes</td><td>5.7-6.2</td><td>39-44</td></tr>
                <tr><td>Diabetes</td><td>&ge;6.3</td><td>&ge;45</td></tr>
            </table>
        </div>

        {% if error %}
        <div id="errorDiv" class="error">
            {{ error }}
        </div>
        {% endif %}
    </div>
    <script>
        function showLoading() {
            document.getElementById('loadingSpinner').classList.remove('hidden');
            document.querySelector('button[type="submit"]').disabled = true;
        }

        function resetResults() {
            document.getElementById('analysisForm').reset();
            const resultDiv = document.getElementById('resultDiv');
            const explanationDiv = document.getElementById('explanationDiv');
            const diagnosisTableDiv = document.getElementById('diagnosisTableDiv');
            const errorDiv = document.getElementById('errorDiv');
            const loadingSpinner = document.getElementById('loadingSpinner');
            if (resultDiv) resultDiv.classList.add('hidden');
            if (explanationDiv) explanationDiv.classList.add('hidden');
            if (diagnosisTableDiv) diagnosisTableDiv.classList.add('hidden');
            if (errorDiv) errorDiv.classList.add('hidden');
            if (loadingSpinner) loadingSpinner.classList.add('hidden');
            document.querySelectorAll('.error-message').forEach(span => span.style.display = 'none');
            validateForm();
        }

        function toggleGuideline(header) {
            const table = header.nextElementSibling;
            table.classList.toggle('visible');
        }

        function validateForm() {
            const fileInput = document.getElementById('file');
            const heartDisease = document.getElementById('heart_disease');
            const smokingHistory = document.getElementById('smoking_history');
            const submitBtn = document.getElementById('submitBtn');

            const fileError = document.getElementById('file-error');
            const heartDiseaseError = document.getElementById('heart-disease-error');
            const smokingHistoryError = document.getElementById('smoking-history-error');

            const isFileValid = fileInput.files.length > 0;
            const isHeartDiseaseValid = heartDisease.value !== '';
            const isSmokingHistoryValid = smokingHistory.value !== '';

            fileError.style.display = isFileValid ? 'none' : 'block';
            heartDiseaseError.style.display = isHeartDiseaseValid ? 'none' : 'block';
            smokingHistoryError.style.display = isSmokingHistoryValid ? 'none' : 'block';

            submitBtn.disabled = !(isFileValid && isHeartDiseaseValid && isSmokingHistoryValid);
        }

        document.getElementById('file').addEventListener('change', validateForm);
        document.getElementById('heart_disease').addEventListener('change', validateForm);
        document.getElementById('smoking_history').addEventListener('change', validateForm);

        validateForm();
    </script>
</body>
</html>
"""

# ==========================
# Routes
# ==========================
@app.route("/", methods=["GET", "POST"])
def home():
    result, color, explanation, error = None, None, None, None
    diagnosis = {}

    if request.method == "POST":
        if 'file' not in request.files:
            error = "No file part"
            return render_template_string(html_page, result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis)

        file = request.files['file']
        if file.filename == '':
            error = "No selected file"
            return render_template_string(html_page, result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis)

        heart_disease = request.form.get('heart_disease')
        smoking_history = request.form.get('smoking_history')

        if not heart_disease or not smoking_history:
            error = "Please select options for Heart Disease and Smoking History."
            return render_template_string(html_page, result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis)

        if file and allowed_file(file.filename):
            file.seek(0, 2)
            file_size = file.tell()
            if file_size > app.config['MAX_CONTENT_LENGTH']:
                error = "File too large. Maximum size is 10MB."
                return render_template_string(html_page, result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis)
            file.seek(0)  # Reset file pointer

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            logger.info(f"Processing file: {filename}")

            try:
                # Extract text from PDF
                text = ''
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + '\n'

                if not text.strip():
                    error = "No text found in the PDF."
                    return render_template_string(html_page, result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis)

                # Add user inputs to diagnosis
                diagnosis['heart_disease'] = {
                    'value': '', 
                    'unit': '', 
                    'status': heart_disease, 
                    'color': 'red' if heart_disease == 'Yes' else 'green'
                }
                diagnosis['smoking_history'] = {
                    'value': '', 
                    'unit': '', 
                    'status': smoking_history.replace('not current', 'Not Current').replace('No Info', 'No Info').capitalize(), 
                    'color': 'black' if smoking_history == 'No Info' else 'green' if smoking_history == 'never' else 'orange' if smoking_history in ['former', 'not current'] else 'red'
                }

                # Parse age
                age_match = re.search(r'(age|Age|AGE)\s*[:=]?\s*(\d+\.?\d*)\s*(years|Years|YEARS)?', text, re.IGNORECASE)
                if age_match:
                    age_value = float(age_match.group(2))
                    diagnosis['age'] = {'value': age_value, 'unit': 'Years', 'status': '', 'color': ''}

                # Parse gender (sex)
                sex_match = re.search(r'(sex|Sex|SEX|gender|Gender|GENDER)\s*[:=]?\s*(male|female|m|f)', text, re.IGNORECASE)
                if sex_match:
                    sex_value = sex_match.group(2).lower().capitalize()
                    if sex_value in ['m', 'male']:
                        sex_value = 'Male'
                    elif sex_value in ['f', 'female']:
                        sex_value = 'Female'
                    diagnosis['sex'] = {'value': sex_value, 'unit': '', 'status': ' ', 'color': 'black'}

                # Parse BMI
                bmi_match = re.search(r'(bmi|BMI|Body Mass Index)\s*[:=]?\s*(\d+\.?\d*)\s*(kg/m²|kg/sqm)?', text, re.IGNORECASE)
                if bmi_match:
                    bmi_value = float(bmi_match.group(2))
                    diagnosis['bmi'] = {'value': bmi_value, 'unit': 'kg/m²', 'status': '', 'color': ''}
                    if bmi_value < 18.5:
                        diagnosis['bmi']['status'] = 'Underweight'
                        diagnosis['bmi']['color'] = 'orange'
                    elif 18.5 <= bmi_value <= 24.9:
                        diagnosis['bmi']['status'] = 'Normal'
                        diagnosis['bmi']['color'] = 'green'
                    elif 25.0 <= bmi_value <= 29.9:
                        diagnosis['bmi']['status'] = 'Pre-obese'
                        diagnosis['bmi']['color'] = 'orange'
                    elif 30.0 <= bmi_value <= 34.9:
                        diagnosis['bmi']['status'] = 'Obese class I'
                        diagnosis['bmi']['color'] = 'red'
                    elif 35.0 <= bmi_value <= 39.9:
                        diagnosis['bmi']['status'] = 'Obese class II'
                        diagnosis['bmi']['color'] = 'red'
                    else:
                        diagnosis['bmi']['status'] = 'Obese class III'
                        diagnosis['bmi']['color'] = 'red'

                # Parse Hypertension (Blood Pressure)
                bp_match = re.search(r'(blood pressure|Blood Pressure|BLOOD PRESSURE)\s*[:=]?\s*(\d+)\s*/\s*(\d+)\s*mmHg', text, re.IGNORECASE)
                if bp_match:
                    systolic = int(bp_match.group(2))
                    diastolic = int(bp_match.group(3))
                    bp_value = f"{systolic}/{diastolic}"
                    diagnosis['hypertension'] = {'value': bp_value, 'unit': 'mmHg', 'status': '', 'color': ''}
                    if systolic < 120 and diastolic < 80:
                        diagnosis['hypertension']['status'] = 'Normal'
                        diagnosis['hypertension']['color'] = 'green'
                    elif (120 <= systolic <= 139 or 80 <= diastolic <= 89):
                        diagnosis['hypertension']['status'] = 'Elevated/Prehypertension'
                        diagnosis['hypertension']['color'] = 'orange'
                    elif (140 <= systolic <= 159 or 90 <= diastolic <= 99):
                        diagnosis['hypertension']['status'] = 'Hypertension Stage 1'
                        diagnosis['hypertension']['color'] = 'red'
                    else:
                        diagnosis['hypertension']['status'] = 'Hypertension Stage 2'
                        diagnosis['hypertension']['color'] = 'red'

                # Find specimen type for glucose
                specimen_match = re.search(r'(Specimen Type|specimen type|fasting|Fasting|normal|NORMAL)\s*[:=]?\s*(fasting|random|normal)', text, re.IGNORECASE)
                specimen_category = None
                if specimen_match:
                    specimen_category = specimen_match.group(2).capitalize()
                    if specimen_category == 'Normal':
                        specimen_category = 'Random'  # Map 'Normal' to 'Random' for glucose context
                else:
                    specimen_category = 'Random'  # Default to Random per request

                # Parse blood glucose
                glucose_matches = re.findall(
                    r'(fasting|random)?\s*(?:glucose|blood sugar|fasting blood glucose|fbs|random blood glucose|rbs)\s*[:=]?\s*(\d+\.?\d*)\s*(mmol/l|mmol/L|mg/dl|mg/dL)?',
                    text, re.IGNORECASE
                )
                if glucose_matches:
                    glucose_context, glucose_str, unit_str = glucose_matches[0]
                    original_value = float(glucose_str)
                    original_unit = unit_str.lower() if unit_str else 'mmol/l'  # Default to mmol/L

                    if 'mg/dl' in original_unit:
                        glucose_value_mmol = original_value / 18.0
                        glucose_value_mg = original_value
                    else:
                        glucose_value_mmol = original_value
                        glucose_value_mg = original_value * 18.0

                    glucose_context = (glucose_context or '').lower()
                    category = specimen_category or (glucose_context.capitalize() if glucose_context else 'Random')

                    diagnosis['glucose'] = {
                        'value_mmol': glucose_value_mmol, 
                        'value_mg': glucose_value_mg, 
                        'category': category,
                        'status': '',
                        'color': ''
                    }

                    if category == 'Fasting':
                        if glucose_value_mmol < 3.9:
                            diagnosis['glucose']['status'] = 'Hypoglycemia'
                            diagnosis['glucose']['color'] = 'red'
                        elif 3.9 <= glucose_value_mmol <= 6.0:
                            diagnosis['glucose']['status'] = 'Normal'
                            diagnosis['glucose']['color'] = 'green'
                        elif 6.1 <= glucose_value_mmol <= 6.9:
                            diagnosis['glucose']['status'] = 'Prediabetes'
                            diagnosis['glucose']['color'] = 'orange'
                        else:
                            diagnosis['glucose']['status'] = 'Diabetes'
                            diagnosis['glucose']['color'] = 'red'
                    else:  # Random (including default)
                        if glucose_value_mmol < 3.9:
                            diagnosis['glucose']['status'] = 'Hypoglycemia'
                            diagnosis['glucose']['color'] = 'red'
                        elif 3.9 <= glucose_value_mmol <= 7.7:
                            diagnosis['glucose']['status'] = 'Normal'
                            diagnosis['glucose']['color'] = 'green'
                        elif 7.8 <= glucose_value_mmol <= 11.0:
                            diagnosis['glucose']['status'] = 'Prediabetes'
                            diagnosis['glucose']['color'] = 'orange'
                        else:
                            diagnosis['glucose']['status'] = 'Diabetes'
                            diagnosis['glucose']['color'] = 'red'

                # Parse HbA1c
                hba1c_matches = re.findall(
                    r'(?:hba1c|a1c|glycosylated hemoglobin)\s*[:=]?\s*(\d+\.?\d*)\s*(%|mmol/mol|mmol/MOL)?',
                    text, re.IGNORECASE
                )
                if hba1c_matches:
                    hba1c_str, unit_str = hba1c_matches[0]
                    original_value = float(hba1c_str)
                    original_unit = unit_str.lower() if unit_str else '%'  # Default to %

                    if original_unit == '%':
                        hba1c_value_percent = original_value
                        hba1c_value_mmol = round((original_value * 10.93) - 23.5)
                    else:
                        hba1c_value_mmol = original_value
                        hba1c_value_percent = round((original_value + 23.5) / 10.93, 1)

                    diagnosis['hba1c'] = {
                        'value_percent': hba1c_value_percent, 
                        'value_mmol': hba1c_value_mmol,
                        'status': '',
                        'color': ''
                    }

                    if hba1c_value_percent < 5.7:
                        diagnosis['hba1c']['status'] = 'Normal'
                        diagnosis['hba1c']['color'] = 'green'
                    elif 5.7 <= hba1c_value_percent <= 6.2:
                        diagnosis['hba1c']['status'] = 'Prediabetes'
                        diagnosis['hba1c']['color'] = 'orange'
                    else:
                        diagnosis['hba1c']['status'] = 'Diabetes'
                        diagnosis['hba1c']['color'] = 'red'

                # Check for required fields for model prediction
                required_fields = ['age', 'sex', 'bmi', 'hypertension', 'glucose', 'hba1c']
                missing = [field for field in required_fields if field not in diagnosis]
                if missing:
                    error = f"Missing required data in the report for prediction: {', '.join(missing)}. Please ensure the PDF contains age, gender, BMI, blood pressure, blood glucose level, and HbA1c level."
                    return render_template_string(html_page, result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis)

                # Prepare input for model
                gender = diagnosis['sex']['value']
                age = float(diagnosis['age']['value'])
                hypertension_val = 1 if diagnosis['hypertension']['status'] in ['Hypertension Stage 1', 'Hypertension Stage 2'] else 0
                heart_disease_val = 1 if heart_disease == 'Yes' else 0
                bmi = diagnosis['bmi']['value']
                HbA1c_level = diagnosis['hba1c']['value_percent']
                blood_glucose_level = diagnosis['glucose']['value_mg']

                input_df = pd.DataFrame({
                    'gender': [gender],
                    'age': [age],
                    'hypertension': [hypertension_val],
                    'heart_disease': [heart_disease_val],
                    'smoking_history': [smoking_history],
                    'bmi': [bmi],
                    'HbA1c_level': [HbA1c_level],
                    'blood_glucose_level': [blood_glucose_level]
                })
                logger.debug(f"Input DataFrame: {input_df.to_dict()}")

                # Preprocess input
                input_preprocessed = preprocessor.transform(input_df)
                logger.debug(f"Preprocessed input shape: {input_preprocessed.shape}")

                # Predict
                sample_pred_proba = model.predict(input_preprocessed, verbose=0)
                sample_pred = (sample_pred_proba > 0.5).astype(int)[0]
                prob = sample_pred_proba[0][0] * 100
                risk = "Diabetes" if sample_pred == 1 else "Normal"
                logger.debug(f"Prediction: {risk}, Probability: {prob:.1f}%")

                # Format result and explanation
                result = f"Prediction: {risk} (Probability of Diabetes: {prob:.1f}%)"
                color = "red" if sample_pred == 1 else "green"
                explanation = "Recommendation: If >50%, consult a doctor!<br>"
                if sample_pred == 1:
                    explanation += f"Flagged as Diabetes due to high HbA1c_level or blood_glucose_level."
                else:
                    explanation += "All vitals within normal ranges. Maintain healthy lifestyle."

            except Exception as e:
                logger.error(f"Error during processing: {str(e)}")
                error = f"Error processing PDF or prediction: {str(e)}"
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)
        else:
            error = "Invalid file type. Please upload a PDF."

        # Save for streaming later
        global latest_prompt
        latest_prompt = f"""
            You are a medical assistant AI. Analyze the following patient data and provide a clear explanation + lifestyle recommendations in simple language.

        Patient Information:
        - Gender: {gender}
        - Age: {age}
        - BMI: {bmi}
        - Blood Pressure: {diagnosis['hypertension']['value']}
        - HbA1c: {HbA1c_level}%
        - Blood Glucose: {blood_glucose_level} mg/dL
        - Heart Disease: {heart_disease}
        - Smoking History: {smoking_history}

        Prediction result: {risk} (Probability: {prob:.1f}%)

        Based on Malaysian clinical guidelines, explain the risk status and give personalized health advice.
        """

    return render_template_string(html_page, result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis)


@app.route('/stream_recommendation')
def stream_recommendation():
    def generate():
        global latest_prompt
        if not latest_prompt:
            yield f"data: {json.dumps({'type': 'error', 'text': 'No prompt available. Please upload a report first.'})}\n\n"
            return

        try:
            for chunk in ollama.chat(
                model="llama3", 
                messages=[{"role": "user", "content": latest_prompt}],
                stream=True,
            ):
                if "message" in chunk and "content" in chunk["message"]:
                    word = chunk["message"]["content"]
                    yield f"data: {json.dumps({'type': 'result', 'text': word})}\n\n"

            # End of stream
            yield f"data: {json.dumps({'type': 'end'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")



if __name__ == "__main__":
    app.run(debug=True)

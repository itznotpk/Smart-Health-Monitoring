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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
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
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #A8D5BA 0%, #C8E6C9 50%, #E8F5E8 100%);
            min-height: 100vh;
            color: #2D3748;
            line-height: 1.6;
        }

        .main-container {
            display: flex;
            min-height: 100vh;
        }

        .sidebar {
            width: 80px;
            background: rgba(45, 55, 72, 0.95);
            backdrop-filter: blur(10px);
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem 0;
            position: fixed;
            height: 100vh;
            right: 0;
            z-index: 1000;
            border-radius: 20px 0 0 20px;
        }

        .sidebar-icon {
            width: 48px;
            height: 48px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            color: white;
            font-size: 20px;
        }

        .sidebar-icon:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }

        .main-content {
            flex: 1;
            padding: 2rem;
            margin-right: 80px;
        }

        .header {
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2rem;
            font-weight: 600;
            color: #2D3748;
            margin-bottom: 0.5rem;
        }

        .header-subtitle {
            color: #718096;
            font-size: 1rem;
        }

        .tab-navigation {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
            background: rgba(255, 255, 255, 0.3);
            padding: 0.5rem;
            border-radius: 16px;
            backdrop-filter: blur(10px);
        }

        .tab-button {
            flex: 1;
            background: transparent;
            border: none;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            color: #4A5568;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .tab-button.active {
            background: linear-gradient(135deg, #4FD1C7 0%, #38B2AC 100%);
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(79, 209, 199, 0.4);
        }

        .tab-button:hover:not(.active) {
            background: rgba(255, 255, 255, 0.5);
            transform: translateY(-1px);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .submit-button {
            background: linear-gradient(135deg, #4FD1C7 0%, #38B2AC 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9rem;
            margin-bottom: 2rem;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }

        .submit-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(79, 209, 199, 0.4);
        }

        .submit-button:disabled {
            background: #CBD5E0;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .result-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .result-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .result-icon {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: white;
        }

        .result-icon.success {
            background: linear-gradient(135deg, #68D391 0%, #38A169 100%);
        }

        .result-icon.warning {
            background: linear-gradient(135deg, #F6AD55 0%, #ED8936 100%);
        }

        .result-icon.danger {
            background: linear-gradient(135deg, #FC8181 0%, #E53E3E 100%);
        }

        .result-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #2D3748;
        }

        .result-subtitle {
            font-size: 1.25rem;
            font-weight: 600;
            color: #4A5568;
        }

        .result-probability {
            font-size: 1rem;
            color: #718096;
            margin-top: 0.5rem;
        }

        .breakdown-section {
            margin-bottom: 2rem;
        }

        .breakdown-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #2D3748;
            margin-bottom: 1rem;
        }

        .breakdown-items {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .breakdown-item {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.6);
            border-radius: 12px;
            transition: all 0.3s ease;
        }

        .breakdown-item:hover {
            background: rgba(255, 255, 255, 0.8);
            transform: translateY(-1px);
        }

        .breakdown-icon {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            color: white;
            background: linear-gradient(135deg, #4FD1C7 0%, #38B2AC 100%);
            flex-shrink: 0;
        }

        .breakdown-content h4 {
            font-weight: 600;
            color: #2D3748;
            margin-bottom: 0.25rem;
        }

        .breakdown-content p {
            color: #718096;
            font-size: 0.9rem;
        }

        .ai-suggestions {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }

        .ai-suggestions h3 {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1rem;
            font-weight: 600;
            color: #2D3748;
            margin-bottom: 1rem;
        }

        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }

        .chart-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }

        .chart-card h3 {
            font-size: 1rem;
            font-weight: 600;
            color: #2D3748;
            margin-bottom: 1rem;
            text-align: center;
        }

        .chart-container {
            position: relative;
            height: 200px;
        }

        .chart-container.large {
            height: 300px;
        }

        .form-section {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        }

        .form-group {
            margin-bottom: 1.5rem;
        }

        .form-group label {
            display: block;
            font-weight: 600;
            color: #2D3748;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }

        .form-group input[type="file"],
        .form-group select {
            width: 100%;
            padding: 0.75rem;
            border: 2px solid #E2E8F0;
            border-radius: 8px;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            background: white;
        }

        .form-group input[type="file"]:focus,
        .form-group select:focus {
            outline: none;
            border-color: #4FD1C7;
            box-shadow: 0 0 0 3px rgba(79, 209, 199, 0.1);
        }

        .error-message {
            color: #E53E3E;
            font-size: 0.8rem;
            margin-top: 0.25rem;
            display: none;
        }

        .disclaimer {
            background: rgba(255, 255, 255, 0.9);
            border-left: 4px solid #4FD1C7;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            font-size: 0.9rem;
            color: #4A5568;
        }

        .table-section {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            overflow-x: auto;
        }

        .table-section table {
            width: 100%;
            border-collapse: collapse;
        }

        .table-section th,
        .table-section td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #E2E8F0;
            font-size: 0.9rem;
        }

        .table-section th {
            background: #F7FAFC;
            font-weight: 600;
            color: #2D3748;
        }

         .guidelines-container {
            display: flex;
            flex-wrap: nowrap;
            gap: 1rem;
            margin-bottom: 2rem;
            justify-content: space-between;
        }

        .guideline-section {
            flex: 1;
            min-width: 0;
            max-width: 24%;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }

        .guideline-header {
            cursor: pointer;
            padding: 0.75rem;
            background: linear-gradient(135deg, #2D3748 0%, #4A5568 100%);
            color: #FFFFFF;
            border-radius: 8px;
            font-weight: 600;
            text-align: center;
            transition: all 0.3s ease;
            margin: 0 0 0.5rem 0;
            font-size: 0.9rem;
        }

        .guideline-header:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(45, 55, 72, 0.4);
        }

        .guideline-table {
            max-height: 0;
            opacity: 0;
            overflow: hidden;
            transition: max-height 0.4s ease, opacity 0.3s ease;
            border-collapse: collapse;
            width: 100%;
            font-size: 0.8rem;
        }

        .guideline-table.visible {
            max-height: 400px;
            opacity: 1;
            margin-top: 0.5rem;
        }

        .guideline-table th,
        .guideline-table td {
            padding: 0.5rem;
            text-align: left;
            border-bottom: 1px solid #E2E8F0;
            font-size: 0.8rem;
        }

        .guideline-table th {
            background: #F7FAFC;
            font-weight: 600;
            color: #2D3748;
        }

        @media (max-width: 768px) {
            .guidelines-container {
                flex-direction: column;
                gap: 0.5rem;
            }
            .guideline-section {
                max-width: 100%;
            }
        }

        .spinner {
            display: inline-block;
            border: 4px solid #E2E8F0;
            border-top: 4px solid #4FD1C7;
            border-radius: 50%;
            width: 32px;
            height: 32px;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .hidden {
            display: none !important;
        }

        .green { color: #38A169; }
        .red { color: #E53E3E; }
        .orange { color: #ED8936; }
        .black { color: #2D3748; }

        .null-cell {
            background-color: #F7FAFC;
            color: #A0AEC0;
        }

        .results-section {
            margin-bottom: 2rem;
        }

        .result {
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            font-weight: 600;
        }

        .result.green {
            background: rgba(104, 211, 145, 0.1);
            border-left: 4px solid #38A169;
            color: #2D3748;
        }

        .result.red {
            background: rgba(229, 62, 62, 0.1);
            border-left: 4px solid #E53E3E;
            color: #2D3748;
        }

        .result.orange {
            background: rgba(237, 137, 54, 0.1);
            border-left: 4px solid #ED8936;
            color: #2D3748;
        }

        #ollama-output {
            white-space: pre-wrap;
            border: 1px solid #E2E8F0;
            padding: 1rem;
            border-radius: 12px;
            margin-top: 1rem;
            background: rgba(255, 255, 255, 0.9);
            min-height: 60px;
            font-size: 0.9rem;
            line-height: 1.6;
        }

        .performance-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            transition: transform 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-4px);
        }

        .metric-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }

        .metric-icon {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            color: white;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #2D3748;
            margin-bottom: 0.25rem;
        }

        .metric-label {
            font-size: 0.9rem;
            color: #718096;
            font-weight: 500;
        }

        .trend-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.5rem;
            font-size: 0.8rem;
        }

        .trend-up {
            color: #38A169;
        }

        .trend-down {
            color: #E53E3E;
        }

        .trend-stable {
            color: #718096;
        }

        .analytics-section {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        }

        .analytics-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #2D3748;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        @media (max-width: 768px) {
            .main-content {
                margin-right: 0;
                padding: 1rem;
            }
            
            .sidebar {
                display: none;
            }
            
            .charts-grid,
            .performance-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 1.5rem;
            }

            .tab-navigation {
                flex-direction: column;
                gap: 0.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="main-content">
            <div class="header">
                <img src="/static/mycaring_logo.png" alt="MyCaring Logo" style="display:block; margin:0; max-width:180px; position:relative; top:0; z-index:1;">
                <h1 style="margin-top:-20px; position:relative; z-index:2;">Hello, Patient! Your Health Report</h1>
                <p class="header-subtitle" style="margin-top:-10px; position:relative; z-index:2;">MyCaringBot - Health History Analysis</p>
            </div>
            <!-- Tab Navigation -->
            <div class="tab-navigation">
                <button class="tab-button active" onclick="switchTab('report-tab', this)">
                    📊 Health Report
                </button>
                <button class="tab-button" onclick="switchTab('performance-tab', this)">
                    📈 Performance Analysis
                </button>
            </div>

            <!-- Report Tab Content -->
            <div id="report-tab" class="tab-content active">
                <div class="disclaimer">
                    <strong>Disclaimer:</strong> This tool is for informational purposes only and uses Malaysian clinical guidelines. Consult a doctor for a proper diagnosis.
                </div>

                <form method="post" enctype="multipart/form-data" id="analysisForm" onsubmit="handleFormSubmission(event)">
                    <div class="form-section">
                        <div class="submit-button-container" style="margin-bottom: 2rem;">
                            <button type="submit" id="submitBtn" class="submit-button" disabled>
                                <span>+</span> Submit a New Report
                            </button>
                        </div>

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
                    </div>
                </form>

                <div id="loadingSpinner" class="loading-spinner hidden">
                    Processing your report...
                    <span class="spinner"></span>
                </div>

                <!-- Results Section -->
                <div id="resultsSection" class="results-section">

                    <!-- Result Message -->
                    {% if result %}
                    <div id="resultDiv" class="result {{ color }}" role="alert">
                        {{ result }}
                    </div>

                    <!-- AI Recommendations -->
                    <div class="ai-suggestions">
                        <h3><span style="font-size: 1.2rem;">💡</span> AI Recommendations</h3>
                        <div id="ollama-output" style="white-space: pre-wrap; border: 1px solid #ccc; padding: 10px; margin-top: 15px;"
                             role="region" aria-live="polite" aria-label="AI Recommendations">
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
                                    outputDiv.innerHTML += data.text;
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

                        document.addEventListener("DOMContentLoaded", function() {
                            {% if result %}
                                startStreaming();
                            {% endif %}
                            {% if diagnosis %}
                                // Pass diagnosis data to chart init
                                const diagnosisData = {{ diagnosis|tojson|safe }};
                                initializeReportCharts(diagnosisData);
                            {% endif %}
                        });
                    </script>
                    {% endif %}

                    <!-- Charts Section -->
                    {% if diagnosis %}
                    <div id="chartsSection" class="charts-grid">
                        <div class="chart-card">
                            <h3>Vital Signs Overview</h3>
                            <div class="chart-container" role="region" aria-label="Vital Signs Visualization">
                                <canvas id="vitalsChart"></canvas>
                            </div>
                        </div>
                        <div class="chart-card">
                            <h3>Prediction Confidence</h3>
                            <div class="chart-container" role="region" aria-label="Diabetes Risk Confidence">
                                <canvas id="confidenceChart"></canvas>
                            </div>
                        </div>
                        <div class="chart-card">
                            <h3>Health Risk Profile</h3>
                            <div class="chart-container large" role="region" aria-label="Health Risk Profile">
                                <canvas id="riskProfileChart"></canvas>
                            </div>
                        </div>
                        <div class="chart-card">
                            <h3>Risk Factor Contribution</h3>
                            <div class="chart-container" role="region" aria-label="Risk Factor Contribution">
                                <canvas id="riskFactorChart"></canvas>
                            </div>
                        </div>
                    </div>

                    <!-- Diagnosis Table -->
                    <div id="diagnosisTableDiv" class="table-section">
                        <table id="diagnosisTable">
                            <tr><th>Metric</th><th>Value</th><th>Unit</th><th>Status</th></tr>
                            {% for key, metric in diagnosis.items() %}
                            <tr>
                                <td>
                                    {% if key == 'heart_disease' %}
                                        Heart Diseases
                                    {% elif key == 'smoking_history' %}
                                        Smoking History
                                    {% else %}
                                        {{ metric.name if metric.name else key|capitalize|replace('_', ' ') }}
                                    {% endif %}
                                </td>
                                <td>{{ metric.value }}</td>
                                <td>{{ metric.unit }}</td>
                                <td class="{{ metric.color }}">{{ metric.status }}</td>
                            </tr>
                            {% endfor %}
                        </table>
                    </div>
                    {% endif %}

                    <!-- Guidelines Section -->
                    <div class="main-container">
                        <div class="main-content">
                            <!-- ... (keep all existing content up to guidelines-container) -->
                            <div class="guidelines-container">
                                <div class="guideline-section">
                                    <h3 class="guideline-header" onclick="toggleGuideline(this)">Hypertension Benchmarks (mmHg)</h3>
                                    <table class="guideline-table">
                                        <tr><th>Status</th><th>Range</th></tr>
                                        <tr><td>Normal</td><td>&lt;120/&lt;80</td></tr>
                                        <tr><td>Elevated/Prehypertension</td><td>120-139/80-89</td></tr>
                                        <tr><td>Hypertension Stage 1</td><td>140-159/90-99</td></tr>
                                        <tr><td>Hypertension Stage 2</td><td>&ge;160/&ge;100</td></tr>
                                    </table>
                                </div>
                                <div class="guideline-section">
                                    <h3 class="guideline-header" onclick="toggleGuideline(this)">Glucose Benchmarks (mmol/L)</h3>
                                    <table class="guideline-table">
                                        <tr><th>Category</th><th>Normal</th><th>Prediabetes</th><th>T2DM Diagnosis</th></tr>
                                        <tr><td>Fasting</td><td>3.9-6.0</td><td>6.1-6.9</td><td>&ge;7.0</td></tr>
                                        <tr><td>Random</td><td>3.9-7.7</td><td>7.8-11.0</td><td>&ge;11.1</td></tr>
                                    </table>
                                </div>
                                <div class="guideline-section">
                                    <h3 class="guideline-header" onclick="toggleGuideline(this)">HbA1c Benchmarks</h3>
                                    <table class="guideline-table">
                                        <tr><th>Status</th><th>%</th><th>mmol/mol</th></tr>
                                        <tr><td>Normal</td><td>&lt;5.7</td><td>&lt;39</td></tr>
                                        <tr><td>Prediabetes</td><td>5.7-6.2</td><td>39-44</td></tr>
                                        <tr><td>Diabetes</td><td>&ge;6.3</td><td>&ge;45</td></tr>
                                    </table>
                                </div>
                                <div class="guideline-section">
                                    <h3 class="guideline-header" onclick="toggleGuideline(this)">BMI Benchmarks (kg/m²)</h3>
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
                            </div>
                        </div>
                    </div>

                    <!-- Add other guideline tables similarly -->

                </div> <!-- end resultsSection -->

            </div> <!-- end report-tab -->


            <!-- Performance Analysis Tab Content -->
            <div id="performance-tab" class="tab-content">
                <div class="analytics-section">
                    <div class="analytics-title">
                        📊 Health Trends Overview
                    </div>
                    
                    <div class="performance-grid">
                        <div class="metric-card">
                            <div class="metric-header">
                                <div class="metric-icon" style="background: linear-gradient(135deg, #68D391, #38A169);">📊</div>
                                <div>
                                    <div class="metric-value">94.8%</div>
                                    <div class="metric-label">Overall Health Score</div>
                                </div>
                            </div>
                            <div class="trend-indicator trend-up">
                                ↗ +2.3% from last month
                            </div>
                        </div>

                        <div class="metric-card">
                            <div class="metric-header">
                                <div class="metric-icon" style="background: linear-gradient(135deg, #4FD1C7, #38B2AC);">🎯</div>
                                <div>
                                    <div class="metric-value">8/10</div>
                                    <div class="metric-label">Health Goals Met</div>
                                </div>
                            </div>
                            <div class="trend-indicator trend-stable">
                                → Consistent performance
                            </div>
                        </div>

                        <div class="metric-card">
                            <div class="metric-header">
                                <div class="metric-icon" style="background: linear-gradient(135deg, #F6AD55, #ED8936);">⚡</div>
                                <div>
                                    <div class="metric-value" id="currentBMI">{{ current_bmi|round(1)|default('--') }}</div>
                                    <div class="metric-label">Current BMI</div>
                                </div>
                            </div>
                            <div class="trend-indicator trend-down">
                                ↘ Improving trend
                            </div>
                        </div>

                        <div class="metric-card">
                            <div class="metric-header">
                                <div class="metric-icon" style="background: linear-gradient(135deg, #9F7AEA, #805AD5);">🔍</div>
                                <div>
                                    <div class="metric-value" id="diabetesRisk">{{ prob|round(1)|default('--') }}%</div>
                                    <div class="metric-label">Diabetes Risk</div>
                                </div>
                            </div>
                            <div class="trend-indicator trend-down">
                                ↘ Risk decreasing
                            </div>
                        </div>
                    </div>
                </div>

                <div class="charts-grid">
                    <div class="chart-card">
                        <h3>Health Score Trends (6 Months)</h3>
                        <div class="chart-container">
                            <canvas id="healthTrendChart"></canvas>
                        </div>
                    </div>
                    <div class="chart-card">
                        <h3>Risk Factors Timeline</h3>
                        <div class="chart-container">
                            <canvas id="riskTimelineChart"></canvas>
                        </div>
                    </div>
                    <div class="chart-card">
                        <h3>Biomarker Progress</h3>
                        <div class="chart-container large">
                            <canvas id="biomarkerChart"></canvas>
                        </div>
                    </div>
                    <div class="chart-card">
                        <h3>Monthly Comparison</h3>
                        <div class="chart-container">
                            <canvas id="monthlyComparisonChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="sidebar">
            <div class="sidebar-icon" title="Menu">☰</div>
            <div class="sidebar-icon" title="Favorites">♡</div>
            <div class="sidebar-icon" title="Calendar">📅</div>
            <div class="sidebar-icon" title="Reports">📋</div>
        </div>
    </div>

    <script>
        // Global variables to store backend data
        let currentDiagnosisData = null;
        let currentPredictionData = null;

        // Tab switching functionality
        function switchTab(tabId, buttonElement) {
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });
            
            document.getElementById(tabId).classList.add('active');
            buttonElement.classList.add('active');
            
            if (tabId === 'performance-tab') {
                setTimeout(initializePerformanceCharts, 100);
            }
        }

        function showLoading() {
            document.getElementById('loadingSpinner').classList.remove('hidden');
            document.querySelector('button[type="submit"]').disabled = true;
        }

        function hideLoading() {
            document.getElementById('loadingSpinner').classList.add('hidden');
            document.querySelector('button[type="submit"]').disabled = false;
        }

        function resetResults() {
            document.getElementById('analysisForm').reset();
            document.getElementById('resultsSection').classList.add('hidden');
            document.getElementById('chartsSection').classList.add('hidden');
            document.getElementById('diagnosisTableDiv').classList.add('hidden');
            document.getElementById('loadingSpinner').classList.add('hidden');
            document.querySelectorAll('.error-message').forEach(span => span.style.display = 'none');
            document.querySelectorAll('.guideline-table').forEach(table => table.classList.remove('visible'));
            validateForm();
        }

        // Enhanced form submission handler that integrates with backend
        function handleFormSubmission(event) {
            // In production, let the form submit to backend
            // In demo mode, simulate the process
            if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
                // Production mode - real backend integration
                showLoading();
                return true; // Allow form submission
            } else {
                // Demo mode - simulate backend response
                event.preventDefault();
                showLoading();
                
                // Simulate backend processing delay
                setTimeout(() => {
                    simulateBackendResponse();
                }, 2000);
                
                return false;
            }
        }

        // Simulate backend response for demo
        function simulateBackendResponse() {
            hideLoading();
            
            // Show results section
            document.getElementById('resultsSection').classList.remove('hidden');
            
            // Simulate result data
            const resultDiv = document.getElementById('resultDiv');
            resultDiv.innerHTML = 'Low Risk of Diabetes - Probability: 5.2%';
            resultDiv.className = 'result green';
            
            // Start AI streaming
            startStreaming();
            
            // Simulate diagnosis data and show charts
            currentDiagnosisData = {
                age: { value: 45, unit: 'years', status: 'Adult', color: 'black' },
                sex: { value: 'Female', unit: '-', status: '-', color: 'black' },
                bmi: { value: 23.5, unit: 'kg/m²', status: 'Normal', color: 'green' },
                heart_disease: { value: 'No', unit: '-', status: 'Good', color: 'green' },
                smoking_history: { value: 'Never', unit: '-', status: 'Excellent', color: 'green' },
                hypertension: { value: 'No', unit: '-', status: 'Normal', color: 'green' },
                glucose: { 
                    category: 'Fasting',
                    value_mg: 95.4, 
                    value_mmol: 5.3, 
                    status: 'Normal', 
                    color: 'green' 
                },
                hba1c: { 
                    value_percent: 5.4, 
                    value_mmol: 36, 
                    status: 'Normal', 
                    color: 'green' 
                }
            };
            
            updateDiagnosisTable(currentDiagnosisData);
            initializeReportCharts(currentDiagnosisData);
            
            // Update performance metrics
            document.getElementById('currentBMI').textContent = '23.5';
            document.getElementById('diabetesRisk').textContent = '5.2%';
        }

        // AI Streaming function from your backend code
        function startStreaming() {
            const eventSource = new EventSource("/stream_recommendation");
            const outputDiv = document.getElementById("ollama-output");
            outputDiv.innerHTML = "";

            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === "result") {
                    outputDiv.innerHTML += data.text;
                } else if (data.type === "end") {
                    eventSource.close();
                } else if (data.type === "error") {
                    outputDiv.innerHTML = "<span style='color:red;'>" + data.text + "</span>";
                    eventSource.close();
                }
            };

            eventSource.onerror = function() {
                // Fallback for demo mode
                if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
                    outputDiv.innerHTML = `
                    Based on your excellent health profile, here are personalized recommendations:

                    🎯 **Maintain Current Success:**
                    • Your BMI of 23.5 is in the optimal range - continue your current lifestyle
                    • HbA1c of 5.4% shows excellent glucose control
                    • Blood glucose levels are perfectly normal

                    🏃‍♀️ **Physical Activity:**
                    • Continue regular physical activity - aim for 150 minutes moderate exercise weekly
                    • Add strength training 2x per week to maintain muscle mass

                    🥗 **Nutrition Guidance:**
                    • Maintain your balanced diet that's keeping your glucose stable
                    • Focus on whole foods, lean proteins, and complex carbohydrates
                    • Stay hydrated with 8 glasses of water daily

                    😴 **Lifestyle Optimization:**
                    • Keep consistent sleep patterns (7-9 hours nightly)
                    • Consider stress management techniques like meditation
                    • Schedule regular health screenings annually

                    ⚠️ **Monitoring:**
                    • Annual diabetes screening recommended
                    • Monitor blood pressure regularly due to family history
                    • Maintain non-smoking status - excellent protective factor
                    `;
                } else {
                    outputDiv.innerHTML += "<br><span style='color:red;'>Error receiving AI response.</span>";
                }
                eventSource.close();
            };
        }

        // Update diagnosis table with backend data
        function updateDiagnosisTable(diagnosis) {
            if (!diagnosis) return;
            
            document.getElementById('diagnosisTableDiv').classList.remove('hidden');
            const table = document.getElementById('diagnosisTable');
            
            // Clear existing rows (except header)
            const rows = table.querySelectorAll('tr:not(:first-child)');
            rows.forEach(row => row.remove());
            
            // Helper function to create table row
            function createRow(label, data, isGlucose = false, isHba1c = false) {
                const row = table.insertRow();
                
                if (data && data.value !== undefined) {
                    if (isGlucose) {
                        // First glucose row (mg/dL)
                        row.innerHTML = `
                            <td>Blood Glucose (${data.category})</td>
                            <td>${Math.round(data.value_mg * 10) / 10}</td>
                            <td>mg/dL</td>
                            <td rowspan="2" class="${data.color}">${data.status}</td>
                        `;
                        // Second glucose row (mmol/L)
                        const row2 = table.insertRow();
                        row2.innerHTML = `
                            <td></td>
                            <td>${Math.round(data.value_mmol * 10) / 10}</td>
                            <td>mmol/L</td>
                        `;
                    } else if (isHba1c) {
                        // First HbA1c row (%)
                        row.innerHTML = `
                            <td>HbA1c Level</td>
                            <td>${Math.round(data.value_percent * 10) / 10}</td>
                            <td>%</td>
                            <td rowspan="2" class="${data.color}">${data.status}</td>
                        `;
                        // Second HbA1c row (mmol/mol)
                        const row2 = table.insertRow();
                        row2.innerHTML = `
                            <td></td>
                            <td>${Math.round(data.value_mmol)}</td>
                            <td>mmol/mol</td>
                        `;
                    } else {
                        row.innerHTML = `
                            <td>${label}</td>
                            <td>${data.value}</td>
                            <td>${data.unit}</td>
                            <td class="${data.color}">${data.status}</td>
                        `;
                    }
                } else {
                    row.innerHTML = `
                        <td>${label}</td>
                        <td class="null-cell">-</td>
                        <td class="null-cell">-</td>
                        <td class="null-cell">-</td>
                    `;
                }
            }
            
            // Create rows for all metrics
            createRow('Age', diagnosis.age);
            createRow('Gender', diagnosis.sex);
            createRow('BMI', diagnosis.bmi);
            createRow('Heart Disease', diagnosis.heart_disease);
            createRow('Smoking History', diagnosis.smoking_history);
            createRow('Hypertension', diagnosis.hypertension);
            if (diagnosis.glucose) createRow('Blood Glucose', diagnosis.glucose, true);
            if (diagnosis.hba1c) createRow('HbA1c Level', diagnosis.hba1c, false, true);
        }

        // Initialize charts with backend data
        function initializeReportCharts(diagnosis) {
            if (!diagnosis) return;
            
            document.getElementById('chartsSection').classList.remove('hidden');
            
            // Vital Signs Chart
            const vitalsCtx = document.getElementById('vitalsChart');
            if (vitalsCtx) {
                new Chart(vitalsCtx, {
                    type: 'bar',
                    data: {
                        labels: ['BMI', 'Blood Glucose (mg/dL)', 'HbA1c (%)'],
                        datasets: [{
                            label: 'Your Values',
                            data: [
                                diagnosis.bmi ? diagnosis.bmi.value : 0,
                                diagnosis.glucose ? diagnosis.glucose.value_mg : 0,
                                diagnosis.hba1c ? diagnosis.hba1c.value_percent : 0
                            ],
                            backgroundColor: ['#68D391', '#68D391', '#68D391'],
                            borderColor: '#2D3748',
                            borderWidth: 2,
                            borderRadius: 8
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } }
                    }
                });
            }

            // Confidence Chart
            const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
            const probMatch = '{{ result }}'.match(/Probability of Diabetes: (\d+\.\d)%/);
            const diabetesProb = probMatch ? parseFloat(probMatch[1]) : 0;
            new Chart(confidenceCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Diabetes Risk', 'Normal'],
                    datasets: [{
                        data: [diabetesProb, 100 - diabetesProb],
                        backgroundColor: ['#FC8181', '#68D391'],
                        borderColor: '#FFFFFF',
                        borderWidth: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '70%',
                    plugins: {
                        legend: { position: 'bottom'}
                    }
                }
            });

            // Risk Profile Chart
            const riskProfileCtx = document.getElementById('riskProfileChart').getContext('2d');
            if (riskProfileCtx) {
                new Chart(riskProfileCtx, {
                    type: 'radar',
                    data: {
                        labels: ['BMI', 'Blood Glucose', 'HbA1c', 'Hypertension', 'Heart Disease', 'Smoking History'],
                        datasets: [{
                            label: 'Your Health Profile',
                            data: [
                                {{ (diagnosis.bmi.value / 40 * 100) | round(1) if 'bmi' in diagnosis else 0 }},
                                {{ (diagnosis.glucose.value_mg / (7.7 * 18.0) * 100) | round(1) if 'glucose' in diagnosis and diagnosis.glucose.category == 'Random' else (diagnosis.glucose.value_mg / (6.0 * 18.0) * 100) | round(1) if 'glucose' in diagnosis else 0 }},
                                {{ (diagnosis.hba1c.value_percent / 6.3 * 100) | round(1) if 'hba1c' in diagnosis else 0 }},
                                {{ 100 if 'hypertension' in diagnosis and diagnosis.hypertension.status in ['Hypertension Stage 1', 'Hypertension Stage 2'] else 0 }},
                                {{ 100 if 'heart_disease' in diagnosis and diagnosis.heart_disease.status == 'Yes' else 0 }},
                                {{ 0 if 'smoking_history' in diagnosis and diagnosis.smoking_history.status == 'Never' else 50 if 'smoking_history' in diagnosis and diagnosis.smoking_history.status in ['Former', 'Not Current'] else 100 if 'smoking_history' in diagnosis else 0 }}
                            ],
                            backgroundColor: 'rgba(79, 209, 199, 0.2)',
                            borderColor: '#4FD1C7',
                            borderWidth: 2,
                            pointBackgroundColor: [
                                '{{ diagnosis.bmi.color if 'bmi' in diagnosis else 'black' }}' === 'green' ? '#9ED048' : '{{ diagnosis.bmi.color if 'bmi' in diagnosis else 'black' }}' === 'orange' ? '#F4A261' : '#D04848',
                                '{{ diagnosis.glucose.color if 'glucose' in diagnosis else 'black' }}' === 'green' ? '#9ED048' : '{{ diagnosis.glucose.color if 'glucose' in diagnosis else 'black' }}' === 'orange' ? '#F4A261' : '#D04848',
                                '{{ diagnosis.hba1c.color if 'hba1c' in diagnosis else 'black' }}' === 'green' ? '#9ED048' : '{{ diagnosis.hba1c.color if 'hba1c' in diagnosis else 'black' }}' === 'orange' ? '#F4A261' : '#D04848',
                                '{{ diagnosis.hypertension.color if 'hypertension' in diagnosis else 'black' }}' === 'green' ? '#9ED048' : '{{ diagnosis.hypertension.color if 'hypertension' in diagnosis else 'black' }}' === 'orange' ? '#F4A261' : '#D04848',
                                '{{ diagnosis.heart_disease.color if 'heart_disease' in diagnosis else 'black' }}' === 'green' ? '#9ED048' : '{{ diagnosis.heart_disease.color if 'heart_disease' in diagnosis else 'black' }}' === 'red' ? '#D04848' : '#F4A261',
                                '{{ diagnosis.smoking_history.color if 'smoking_history' in diagnosis else 'black' }}' === 'green' ? '#9ED048' : '{{ diagnosis.smoking_history.color if 'smoking_history' in diagnosis else 'black' }}' === 'orange' ? '#F4A261' : '{{ diagnosis.smoking_history.color if 'smoking_history' in diagnosis else 'black' }}' === 'red' ? '#D04848' : '#3B3C50'
                            ]
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            r: { beginAtZero: true, max: 100 }
                        },
                        plugins: {
                            legend: { position: 'bottom', labels: { color: '#3B3C50', font: { size: 14 } } },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        const label = context.label;
                                        const value = context.raw;
                                        const statuses = {
                                            'BMI': '{{ diagnosis.bmi.status if 'bmi' in diagnosis else 'Not Available' }}',
                                            'Blood Glucose': '{{ diagnosis.glucose.status if 'glucose' in diagnosis else 'Not Available' }}',
                                            'HbA1c': '{{ diagnosis.hba1c.status if 'hba1c' in diagnosis else 'Not Available' }}',
                                            'Hypertension': '{{ diagnosis.hypertension.status if 'hypertension' in diagnosis else 'Not Available' }}',
                                            'Heart Disease': '{{ diagnosis.heart_disease.status if 'heart_disease' in diagnosis else 'Not Available' }}',
                                            'Smoking History': '{{ diagnosis.smoking_history.status if 'smoking_history' in diagnosis else 'Not Available' }}'
                                        };
                                        return `${label}: ${value}% (Status: ${statuses[label]})`;
                                    }
                                }
                            }
                        }
                    }
                });
            }

            // Risk Factor Contribution Chart - This should come from your ML model feature importance
            const riskFactorCtx = document.getElementById('riskFactorChart');
            if (riskFactorCtx) {
                new Chart(riskFactorCtx, {
                    type: 'pie',
                    data: {
                        labels: ['HbA1c', 'Blood Glucose', 'Age', 'BMI', 'Hypertension'],
                        datasets: [{
                            data: [23.07, 10.50, 2.84, 0.80, 0.16], // These should come from your model
                            backgroundColor: ['#68D391', '#4FD1C7', '#2D3748', '#F6AD55', '#9F7AEA'],
                            borderColor: '#FFFFFF',
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: 'bottom' }
                        }
                    }
                });
            }
        }

        // Initialize Performance Charts
        function initializePerformanceCharts() {
            // Health Trend Chart
            const healthTrendCtx = document.getElementById('healthTrendChart');
            if (healthTrendCtx && !Chart.getChart(healthTrendCtx)) {
                new Chart(healthTrendCtx, {
                    type: 'line',
                    data: {
                        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                        datasets: [{
                            label: 'Health Score',
                            data: [88.5, 89.2, 91.2, 92.1, 93.1, 94.8],
                            borderColor: '#4FD1C7',
                            backgroundColor: 'rgba(79, 209, 199, 0.1)',
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            y: { beginAtZero: false, min: 85 }
                        }
                    }
                });
            }

            // Risk Timeline Chart
            const riskTimelineCtx = document.getElementById('riskTimelineChart');
            if (riskTimelineCtx && !Chart.getChart(riskTimelineCtx)) {
                new Chart(riskTimelineCtx, {
                    type: 'line',
                    data: {
                        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                        datasets: [{
                            label: 'Diabetes Risk %',
                            data: [8.2, 7.8, 6.8, 6.3, 6.0, 5.2],
                            borderColor: '#FC8181',
                            backgroundColor: 'rgba(252, 129, 129, 0.1)',
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            y: { beginAtZero: true, max: 10 }
                        }
                    }
                });
            }

            // Biomarker Progress Chart
            const biomarkerCtx = document.getElementById('biomarkerChart');
            if (biomarkerCtx && !Chart.getChart(biomarkerCtx)) {
                const currentBMI = {{ diagnosis.bmi.value|default(23.5) }};
                const currentHbA1c = {{ diagnosis.hba1c.value_percent|default(5.4) }};
                const currentGlucose = {{ diagnosis.glucose.value_mg|default(95) }};
                const currentHypertension = {{ (100 if diagnosis.hypertension.status in ['Hypertension Stage 1', 'Hypertension Stage 2'] else 0)|default(0) }};
                
                new Chart(biomarkerCtx, {
                    type: 'radar',
                    data: {
                        labels: ['BMI', 'HbA1c', 'Glucose', 'Blood Pressure', 'Physical Activity', 'Diet Quality'],
                        datasets: [
                            {
                                label: 'Current',
                                data: [
                                    {{ (diagnosis.bmi.value / 40 * 100)|round(1)|default(85) }},
                                    {{ (diagnosis.hba1c.value_percent / 6.3 * 100)|round(1)|default(92) }},
                                    {{ (diagnosis.glucose.value_mg / (7.7 * 18.0) * 100)|round(1)|default(88) if diagnosis.glucose.category|default('Random') == 'Random' else (diagnosis.glucose.value_mg / (6.0 * 18.0) * 100)|round(1)|default(88) }},
                                    currentHypertension,
                                    90, // Static for Physical Activity
                                    87  // Static for Diet Quality
                                ],
                                borderColor: '#4FD1C7',
                                backgroundColor: 'rgba(79, 209, 199, 0.2)'
                            },
                            {
                                label: '3 Months Ago',
                                data: [75, 88, 85, 95, 85, 82],
                                borderColor: '#CBD5E0',
                                backgroundColor: 'rgba(203, 213, 224, 0.1)'
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            r: { beginAtZero: true, max: 100 }
                        }
                    }
                });
            }

            // Monthly Comparison Chart
            const monthlyCtx = document.getElementById('monthlyComparisonChart');
            if (monthlyCtx && !Chart.getChart(monthlyCtx)) {
                const currentBMI = {{ diagnosis.bmi.value|default(23.5) }};
                const currentHbA1c = {{ diagnosis.hba1c.value_percent|default(5.4) }};
                const currentGlucose = {{ diagnosis.glucose.value_mg|default(95) }};
                
                new Chart(monthlyCtx, {
                    type: 'bar',
                    data: {
                        labels: ['BMI', 'HbA1c', 'Glucose'],
                        datasets: [
                            {
                                label: 'Last Month',
                                data: [{{ (diagnosis.bmi.value + 0.6)|default(24.1) }}, {{ (diagnosis.hba1c.value_percent + 0.1)|default(5.5) }}, {{ (diagnosis.glucose.value_mg + 1)|default(96) }}],
                                backgroundColor: '#CBD5E0'
                            },
                            {
                                label: 'Current',
                                data: [currentBMI, currentHbA1c, currentGlucose],
                                backgroundColor: '#4FD1C7'
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: 'bottom' }
                        }
                    }
                });
            }
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

        // Event listeners
        document.getElementById('file').addEventListener('change', validateForm);
        document.getElementById('heart_disease').addEventListener('change', validateForm);
        document.getElementById('smoking_history').addEventListener('change', validateForm);

        // Initial validation
        document.addEventListener('DOMContentLoaded', function() {
            validateForm();
        });
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
    prob = 0  # Initialize prob with a default value

    if request.method == "POST":
        if 'file' not in request.files:
            error = "No file part"
            return render_template_string(html_page, result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis, prob=prob)

        file = request.files['file']
        if file.filename == '':
            error = "No selected file"
            return render_template_string(html_page, result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis, prob=prob)

        heart_disease = request.form.get('heart_disease')
        smoking_history = request.form.get('smoking_history')

        if not heart_disease or not smoking_history:
            error = "Please select options for Heart Disease and Smoking History."
            return render_template_string(html_page, result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis, prob=prob)

        if file and allowed_file(file.filename):
            file.seek(0, 2)
            file_size = file.tell()
            if file_size > app.config['MAX_CONTENT_LENGTH']:
                error = "File too large. Maximum size is 10MB."
                return render_template_string(html_page, result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis, prob=prob)
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
                    return render_template_string(html_page, result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis, prob=prob)

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
                    return render_template_string(html_page, result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis, prob=prob)

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
                prob = sample_pred_proba[0][0] * 100  # Assign prob here
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

    return render_template_string(html_page, result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis, prob=prob, current_bmi=bmi_value)

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

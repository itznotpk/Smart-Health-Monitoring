from pydoc import text
from flask import Flask, render_template_string, request, stream_with_context, Response, jsonify
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
# Routes
# ==========================
@app.route("/", methods=["GET", "POST"])
def home():
    result, color, explanation, error = None, None, None, None
    diagnosis = {}

    if request.method == "POST":
        if 'file' not in request.files:
            error = "No file part"
            return render_template_string("index.html", result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis)

        file = request.files['file']
        if file.filename == '':
            error = "No selected file"
            return render_template_string("index.html", result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis)

        heart_disease = request.form.get('heart_disease')
        smoking_history = request.form.get('smoking_history')

        if not heart_disease or not smoking_history:
            error = "Please select options for Heart Disease and Smoking History."
            return render_template_string("index.html", result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis)

        if file and allowed_file(file.filename):
            file.seek(0, 2)
            file_size = file.tell()
            if file_size > app.config['MAX_CONTENT_LENGTH']:
                error = "File too large. Maximum size is 10MB."
                return render_template_string("index.html", result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis)
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
                    return render_template_string("index.html", result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis)

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
                    return render_template_string("index.html", result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis)

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

    return render_template_string("index.html", result=result, color=color, explanation=explanation, error=error, diagnosis=diagnosis)

@app.route("/analyze", methods=["POST"])
def analyze():
    response = {"error": None, "risk_score": None, "explanation": None, "diagnosis": None}

    try:
        # --- Check file & form inputs ---
        if 'file' not in request.files:
            response["error"] = "No file uploaded."
            logger.warning("No file part in request.")
            return jsonify(response)

        file = request.files['file']
        heart_disease = request.form.get('heart_disease')
        smoking_history = request.form.get('smoking_history')

        if not heart_disease or not smoking_history:
            response["error"] = "Please select Heart Disease and Smoking History."
            logger.warning("Heart Disease or Smoking History not provided.")
            return jsonify(response)

        if not (file and allowed_file(file.filename)):
            response["error"] = "Invalid file type. Only PDF allowed."
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify(response)

        # --- Save PDF temporarily ---
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        text = ""
        try:
            # --- Extract text from PDF ---
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if not text.strip():
                response["error"] = "No text found in the PDF."
                logger.warning(f"No text extracted from PDF: {filename}")
                return jsonify(response)
        except Exception as e:
            response["error"] = f"Error reading PDF: {str(e)}"
            logger.error(f"PDF reading error: {str(e)}")
            return jsonify(response)

        diagnosis = {}
        # --- Parsing section with safe defaults ---
        try:
            # Age
            age_match = re.search(r'(age)\s*[:=]?\s*(\d+\.?\d*)', text, re.IGNORECASE)
            age_val = float(age_match.group(2)) if age_match else 30
            diagnosis['age'] = {'value': age_val, 'unit': 'Years'}

            # Sex / Gender
            sex_match = re.search(r'(sex|gender)\s*[:=]?\s*(male|female|m|f|other)', text, re.IGNORECASE)
            if sex_match:
                sex_val = sex_match.group(2).lower()
                if sex_val in ['m', 'male']:
                    sex_val = 'Male'
                elif sex_val in ['f', 'female']:
                    sex_val = 'Female'
                else:
                    sex_val = 'Other'
            else:
                sex_val = 'Male'  # default
            diagnosis['sex'] = {'value': sex_val}

            # BMI
            bmi_match = re.search(r'(bmi|body mass index)\s*[:=]?\s*(\d+\.?\d*)', text, re.IGNORECASE)
            bmi_val = float(bmi_match.group(2)) if bmi_match else 22.0
            diagnosis['bmi'] = {'value': bmi_val}

            # Blood Pressure / Hypertension
            bp_match = re.search(r'(\d+)\s*/\s*(\d+)', text)
            systolic, diastolic = (int(bp_match.group(1)), int(bp_match.group(2))) if bp_match else (120, 80)
            hypertension_val = 1 if systolic >= 140 or diastolic >= 90 else 0
            diagnosis['hypertension'] = {'value': f"{systolic}/{diastolic}",
                                         'status': 'Hypertension' if hypertension_val else 'Normal'}

            # Glucose
            glucose_match = re.search(r'(?:glucose|blood sugar)\s*[:=]?\s*(\d+\.?\d*)', text, re.IGNORECASE)
            glucose_val_mg = float(glucose_match.group(1)) if glucose_match else 90.0
            diagnosis['glucose'] = {'value_mg': glucose_val_mg}

            # HbA1c
            hba1c_match = re.search(r'(?:hba1c|a1c)\s*[:=]?\s*(\d+\.?\d*)', text, re.IGNORECASE)
            hba1c_val = float(hba1c_match.group(1)) if hba1c_match else 5.5
            diagnosis['hba1c'] = {'value_percent': hba1c_val}

        except Exception as e:
            response["error"] = f"Error parsing PDF data: {str(e)}"
            logger.error(f"Parsing error: {str(e)}")
            return jsonify(response)

        # --- Prepare input for preprocessor ---
        try:
            valid_genders = ['Female', 'Male', 'Other']
            valid_smoking = ['No Info', 'current', 'former', 'never', 'not current']

            if sex_val not in valid_genders:
                sex_val = 'Other'
            if smoking_history not in valid_smoking:
                smoking_history = 'No Info'

            input_df = pd.DataFrame({
                'age': [age_val],
                'hypertension': [hypertension_val],
                'heart_disease': [1 if heart_disease == 'Yes' else 0],
                'bmi': [bmi_val],
                'HbA1c_level': [hba1c_val],
                'blood_glucose_level': [glucose_val_mg],
                'gender': [sex_val],
                'smoking_history': [smoking_history]
            })

            # --- Preprocess & Predict ---
            input_preprocessed = preprocessor.transform(input_df)
            pred_proba = model.predict(input_preprocessed, verbose=0)[0][0]
            pred_label = 1 if pred_proba > 0.5 else 0

            risk = "Diabetes" if pred_label else "Normal"
            prob = pred_proba * 100

            response["risk_score"] = f"{risk} ({prob:.1f}%)"
            response["explanation"] = ("Consult a doctor if high risk."
                                       if pred_label else
                                       "All vitals normal. Maintain healthy lifestyle.")
            response["diagnosis"] = diagnosis

        except Exception as e:
            response["error"] = f"Error during preprocessing or prediction: {str(e)}"
            logger.error(f"Prediction error: {str(e)}")

        finally:
            # Always attempt to remove file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove file {file_path}: {str(e)}")

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        logger.error(traceback_str)
        response["error"] = f"Unexpected server error: {str(e)}"

    return jsonify(response)




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

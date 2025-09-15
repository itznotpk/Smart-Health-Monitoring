from flask import Flask, render_template_string, request
import re
import pdfplumber
import os
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, filename='app.log', format='%(asctime)s - %(levelname)s - %(message)s')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ==========================
# Web UI
# ==========================
html_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Diabetes Health Checker from Clinical Report</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; background-color: #f8f9fa; }
        .container { margin-top: 30px; }
        input, select { margin: 8px; padding: 8px; }
        button { padding: 10px 20px; margin-top: 15px; }
        .result { font-size: 22px; margin-top: 20px; font-weight: bold; }
        .explanation { font-size: 16px; margin-top: 10px; }
        .green { color: green; }
        .red { color: red; }
        .orange { color: orange; }
        .black { color: black; }
        table { margin: 20px auto; border-collapse: collapse; width: 80%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .guideline-box { border: 1px solid #ddd; padding: 10px; margin: 10px auto; width: 80%; }
        .null-cell { background-color: #cccccc; }
    </style>
</head>
<body>
    <h1>AI-Powered Diabetes Checker</h1>
    <div class="container">
        <p><strong>Disclaimer:</strong> This tool is for informational purposes only and uses Malaysian clinical guidelines. Consult a doctor for a proper diagnosis.</p>
        <form method="post" enctype="multipart/form-data" onsubmit="showLoading()">
            <label for="file">Upload Clinical Report (PDF):</label><br>
            <input type="file" name="file" accept=".pdf" required><br>
            <label for="heart_diseases">Heart Diseases (past/current):</label><br>
            <select name="heart_diseases" required>
                <option value="">Select</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
            </select><br>
            <label for="smoking_history">Smoking History:</label><br>
            <select name="smoking_history" required>
                <option value="">Select</option>
                <option value="No Info">No Info</option>
                <option value="never">never</option>
                <option value="former">former</option>
                <option value="current">current</option>
                <option value="not current">not current</option>
            </select><br>
            <button type="submit">Analyze Report</button>
        </form>
        <button onclick="location.reload();">Refresh and Reupload</button>

        {% if diagnosis %}
        <div class="result {{ overall_color }}">
            {{ overall_result }}
        </div>
        <table>
            <tr><th>Metric</th><th>Category</th><th>Value</th><th>Unit</th><th>Status</th></tr>
            {% if 'age' in diagnosis %}
            <tr>
                <td>Age</td>
                <td>{{ diagnosis.age.category }}</td>
                <td>{{ diagnosis.age.value }}</td>
                <td>{{ diagnosis.age.unit }}</td>
                <td class="{{ diagnosis.age.color }}">{{ diagnosis.age.status }}</td>
            </tr>
            {% else %}
            <tr><td>Age</td><td class="null-cell">-</td><td class="null-cell">-</td><td class="null-cell">-</td><td class="null-cell">-</td></tr>
            {% endif %}
            {% if 'sex' in diagnosis %}
            <tr>
                <td>Sex</td>
                <td>{{ diagnosis.sex.category }}</td>
                <td>{{ diagnosis.sex.value }}</td>
                <td>{{ diagnosis.sex.unit }}</td>
                <td class="{{ diagnosis.sex.color }}">{{ diagnosis.sex.status }}</td>
            </tr>
            {% else %}
            <tr><td>Sex</td><td class="null-cell">-</td><td class="null-cell">-</td><td class="null-cell">-</td><td class="null-cell">-</td></tr>
            {% endif %}
            {% if 'bmi' in diagnosis %}
            <tr>
                <td>BMI</td>
                <td>{{ diagnosis.bmi.category }}</td>
                <td>{{ diagnosis.bmi.value | round(1) }}</td>
                <td>{{ diagnosis.bmi.unit }}</td>
                <td class="{{ diagnosis.bmi.color }}">{{ diagnosis.bmi.status }}</td>
            </tr>
            {% else %}
            <tr><td>BMI</td><td class="null-cell">-</td><td class="null-cell">-</td><td class="null-cell">-</td><td class="null-cell">-</td></tr>
            {% endif %}
            {% if 'heart_diseases' in diagnosis %}
            <tr>
                <td>Heart Diseases</td>
                <td>{{ diagnosis.heart_diseases.category }}</td>
                <td>-</td>
                <td>{{ diagnosis.heart_diseases.unit }}</td>
                <td class="{{ diagnosis.heart_diseases.color }}">{{ diagnosis.heart_diseases.value }}</td>
            </tr>
            {% else %}
            <tr><td>Heart Diseases</td><td class="null-cell">-</td><td class="null-cell">-</td><td class="null-cell">-</td><td class="null-cell">-</td></tr>
            {% endif %}
            {% if 'smoking_history' in diagnosis %}
            <tr>
                <td>Smoking History</td>
                <td>{{ diagnosis.smoking_history.category }}</td>
                <td>-</td>
                <td>{{ diagnosis.smoking_history.unit }}</td>
                <td class="{{ diagnosis.smoking_history.color }}">{{ diagnosis.smoking_history.value }}</td>
            </tr>
            {% else %}
            <tr><td>Smoking History</td><td class="null-cell">-</td><td class="null-cell">-</td><td class="null-cell">-</td><td class="null-cell">-</td></tr>
            {% endif %}
            {% if 'hypertension' in diagnosis %}
            <tr>
                <td>Hypertension</td>
                <td>{{ diagnosis.hypertension.category }}</td>
                <td>{{ diagnosis.hypertension.value }}</td>
                <td>{{ diagnosis.hypertension.unit }}</td>
                <td class="{{ diagnosis.hypertension.color }}">{{ diagnosis.hypertension.status }}</td>
            </tr>
            {% else %}
            <tr><td>Hypertension</td><td class="null-cell">-</td><td class="null-cell">-</td><td class="null-cell">-</td><td class="null-cell">-</td></tr>
            {% endif %}
            {% if 'glucose' in diagnosis %}
            <tr>
                <td rowspan="2">Glucose</td>
                <td rowspan="2">{{ diagnosis.glucose.category }}</td>
                <td>{{ diagnosis.glucose.value_mmol | round(1) }}</td>
                <td>mmol/L</td>
                <td rowspan="2" class="{{ diagnosis.glucose.color }}">{{ diagnosis.glucose.status }}</td>
            </tr>
            <tr>
                <td>{{ diagnosis.glucose.value_mg | round(1) }}</td>
                <td>mg/dL</td>
            </tr>
            {% endif %}
            {% if 'hba1c' in diagnosis %}
            <tr>
                <td rowspan="2">HbA1c</td>
                <td rowspan="2">-</td>
                <td>{{ diagnosis.hba1c.value_percent | round(1) }}</td>
                <td>%</td>
                <td rowspan="2" class="{{ diagnosis.hba1c.color }}">{{ diagnosis.hba1c.status }}</td>
            </tr>
            <tr>
                <td>{{ diagnosis.hba1c.value_mmol | round(0) }}</td>
                <td>mmol/mol</td>
            </tr>
            {% endif %}
        </table>
        <div class="guideline-box">
            <h3>Hypertension Benchmarks (mmHg)</h3>
            <table>
                <tr><th>Status</th><th>Range</th></tr>
                <tr><td>Normal</td><td>&lt;120/&lt;80</td></tr>
                <tr><td>Elevated/Prehypertension</td><td>120-139/80-89</td></tr>
                <tr><td>Hypertension Stage 1</td><td>140-159/90-99</td></tr>
                <tr><td>Hypertension Stage 2</td><td>&ge;160/&ge;100</td></tr>
            </table>
        </div>
        <div class="guideline-box">
            <h3>BMI Benchmarks (kg/m²)</h3>
            <table>
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
            <h3>Glucose Benchmarks (mmol/L)</h3>
            <table>
                <tr><th>Category</th><th>Normal</th><th>Prediabetes</th><th>T2DM Diagnosis</th></tr>
                <tr><td>Fasting</td><td>3.9-6.0</td><td>6.1-6.9</td><td>&ge;7.0</td></tr>
                <tr><td>Random</td><td>3.9-7.7</td><td>7.8-11.0</td><td>&ge;11.1</td></tr>
            </table>
            <p>T2DM = Type 2 Diabetes Mellitus</p>
        </div>
        <div class="guideline-box">
            <h3>HbA1c Benchmarks</h3>
            <table>
                <tr><th>Status</th><th>%</th><th>mmol/mol</th></tr>
                <tr><td>Normal</td><td>&lt;5.7</td><td>&lt;39</td></tr>
                <tr><td>Prediabetes</td><td>5.7-6.2</td><td>39-44</td></tr>
                <tr><td>Diabetes</td><td>&ge;6.3</td><td>&ge;45</td></tr>
            </table>
        </div>
        {% endif %}

        {% if error %}
        <div class="result red">
            {{ error }}
        </div>
        {% endif %}
    </div>
    <script>
        function showLoading() {
            document.querySelector('button[type="submit"]').innerText = 'Analyzing...';
        }
    </script>
</body>
</html>
"""

# ==========================
# Routes
# ==========================
@app.route("/", methods=["GET", "POST"])
def home():
    diagnosis, overall_result, overall_color, error = {}, None, None, None

    if request.method == "POST":
        if 'file' not in request.files:
            error = "No file part"
            return render_template_string(html_page, error=error)

        file = request.files['file']
        if file.filename == '':
            error = "No selected file"
            return render_template_string(html_page, error=error)

        heart_diseases = request.form.get('heart_diseases')
        smoking_history = request.form.get('smoking_history')

        if not heart_diseases or not smoking_history:
            error = "Please select options for Heart Diseases and Smoking History."
            return render_template_string(html_page, error=error)

        if file and allowed_file(file.filename):
            if len(file.read()) > app.config['MAX_CONTENT_LENGTH']:
                error = "File too large. Maximum size is 10MB."
                return render_template_string(html_page, error=error)
            file.seek(0)  # Reset file pointer after reading length

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            logging.info(f"Processing file: {filename}")

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
                    return render_template_string(html_page, error=error)

                # Add user inputs with status coloring
                diagnosis['heart_diseases'] = {'value': heart_diseases, 'unit': '-', 'category': '-', 'status': '', 'color': 'red' if heart_diseases == 'Yes' else 'green'}
                diagnosis['smoking_history'] = {'value': smoking_history, 'unit': '-', 'category': '-', 'status': '', 'color': 'black' if smoking_history == 'No Info' else 'green' if smoking_history == 'never' else 'orange' if smoking_history in ['former', 'not current'] else 'red'}

                # Parse age
                age_match = re.search(r'(age|Age|AGE)\s*[:=]?\s*(\d+)\s*(years|Years|YEARS)?', text, re.IGNORECASE)
                if age_match:
                    age_value = int(age_match.group(2))
                    diagnosis['age'] = {'value': age_value, 'unit': 'Years', 'category': '-', 'status': '', 'color': ''}

                # Parse sex
                sex_match = re.search(r'(sex|Sex|SEX|gender|Gender|GENDER)\s*[:=]?\s*(male|female|m|f)', text, re.IGNORECASE)
                if sex_match:
                    sex_value = sex_match.group(2).lower()
                    if sex_value in ['m', 'male']:
                        sex_value = 'Male'
                    elif sex_value in ['f', 'female']:
                        sex_value = 'Female'
                    diagnosis['sex'] = {'value': sex_value, 'unit': '-', 'category': '-', 'status': '', 'color': ''}

                # Parse BMI
                bmi_match = re.search(r'(bmi|BMI|Body Mass Index)\s*[:=]?\s*(\d+\.?\d*)\s*(kg/m²|kg/sqm)?', text, re.IGNORECASE)
                if bmi_match:
                    bmi_value = float(bmi_match.group(2))
                    diagnosis['bmi'] = {'value': bmi_value, 'unit': 'kg/m²', 'category': '-', 'status': '', 'color': ''}
                    if bmi_value < 18.5:
                        diagnosis['bmi']['status'] = 'Underweight'
                        diagnosis['bmi']['color'] = 'orange'
                    elif 18.5 <= bmi_value <= 24.9:
                        diagnosis['bmi']['status'] = 'Normal range'
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
                    diagnosis['hypertension'] = {'value': bp_value, 'unit': 'mmHg', 'category': '-', 'status': '', 'color': ''}
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

                # Find specimen type
                specimen_match = re.search(r'Specimen Type\s*[:=]?\s*(fasting|random)', text, re.IGNORECASE)
                specimen_category = specimen_match.group(1).capitalize() if specimen_match else None

                # Parse glucose with context (fasting or random)
                glucose_matches = re.findall(
                    r'(fasting|random)?\s*(?:glucose|blood sugar|fasting blood glucose|fbs|random blood glucose|rbs)\s*[:=]?\s*(\d+\.?\d*)\s*(mmol/l|mmol/L|mg/dl|mg/dL)?',
                    text, re.IGNORECASE
                )
                glucose_context, glucose_value_mmol, glucose_value_mg, category = None, None, None, None
                if glucose_matches:
                    # Take the first match for simplicity; could extend to handle multiple
                    glucose_context, glucose_str, unit_str = glucose_matches[0]
                    original_value = float(glucose_str)
                    original_unit = unit_str.lower() if unit_str else 'mmol/l'  # Default to mmol/L for Malaysia

                    if original_unit in ['mg/dl', 'mg/dl']:
                        glucose_value_mmol = original_value / 18.0
                        glucose_value_mg = original_value
                    else:
                        glucose_value_mmol = original_value
                        glucose_value_mg = original_value * 18.0

                    glucose_context = glucose_context.lower() if glucose_context else None
                    category = specimen_category or glucose_context or 'Unknown'

                    if category != 'Unknown':
                        category = category.capitalize()

                    if glucose_value_mmol is not None:
                        diagnosis['glucose'] = {'value_mmol': glucose_value_mmol, 'value_mg': glucose_value_mg, 'category': category}

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
                        elif category == 'Random':
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
                        else:
                            # Unknown category: Assume fasting with note
                            diagnosis['glucose']['status'] = 'Unknown Category - Assuming Fasting'
                            diagnosis['glucose']['color'] = 'orange'
                            if glucose_value_mmol < 3.9:
                                diagnosis['glucose']['status'] += ' (Hypoglycemia)'
                                diagnosis['glucose']['color'] = 'red'
                            elif 3.9 <= glucose_value_mmol <= 6.0:
                                diagnosis['glucose']['status'] += ' (Normal)'
                            elif 6.1 <= glucose_value_mmol <= 6.9:
                                diagnosis['glucose']['status'] += ' (Prediabetes)'
                            else:
                                diagnosis['glucose']['status'] += ' (Diabetes)'
                                diagnosis['glucose']['color'] = 'red'

                # Parse HbA1c with unit (% or mmol/mol)
                hba1c_matches = re.findall(
                    r'(?:hba1c|a1c|glycosylated hemoglobin)\s*[:=]?\s*(\d+\.?\d*)\s*(%|mmol/mol|mmol/MOL)?',
                    text, re.IGNORECASE
                )
                hba1c_value_percent, hba1c_value_mmol = None, None
                if hba1c_matches:
                    # Take the first match
                    hba1c_str, unit_str = hba1c_matches[0]
                    original_value = float(hba1c_str)
                    original_unit = unit_str.lower() if unit_str else '%'  # Default to %

                    if original_unit == '%':
                        hba1c_value_percent = original_value
                        hba1c_value_mmol = round((original_value * 10.93) - 23.5)
                    else:
                        hba1c_value_mmol = original_value
                        hba1c_value_percent = round((original_value + 23.5) / 10.93, 1)

                    diagnosis['hba1c'] = {'value_percent': hba1c_value_percent, 'value_mmol': hba1c_value_mmol}

                    # Determine status using % ranges (equivalent for mmol)
                    if hba1c_value_percent < 5.7:
                        diagnosis['hba1c']['status'] = 'Normal'
                        diagnosis['hba1c']['color'] = 'green'
                    elif 5.7 <= hba1c_value_percent <= 6.2:
                        diagnosis['hba1c']['status'] = 'Prediabetes'
                        diagnosis['hba1c']['color'] = 'orange'
                    else:
                        diagnosis['hba1c']['status'] = 'Diabetes'
                        diagnosis['hba1c']['color'] = 'red'

                if diagnosis:
                    # Determine overall result based on worst status (only for glucose, hba1c, and hypertension)
                    statuses = []
                    if 'glucose' in diagnosis:
                        statuses.append(diagnosis['glucose']['status'])
                    if 'hba1c' in diagnosis:
                        statuses.append(diagnosis['hba1c']['status'])
                    if 'hypertension' in diagnosis:
                        statuses.append(diagnosis['hypertension']['status'])
                    if any('Diabetes' in s or 'Hypertension Stage' in s for s in statuses):
                        overall_result = 'Diabetes or Hypertension Indicated'
                        overall_color = 'red'
                    elif any('Prediabetes' in s or 'Elevated/Prehypertension' in s for s in statuses):
                        overall_result = 'Prediabetes or Prehypertension Indicated'
                        overall_color = 'orange'
                    else:
                        overall_result = 'Normal'
                        overall_color = 'green'
                else:
                    error = "No diabetes-related or hypertension information found in the report."

            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
                error = f"Error processing PDF: {str(e)}"
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)
        else:
            error = "Invalid file type. Please upload a PDF."

    return render_template_string(html_page, diagnosis=diagnosis, overall_result=overall_result, overall_color=overall_color, error=error)

if __name__ == "__main__":
    app.run(debug=True)

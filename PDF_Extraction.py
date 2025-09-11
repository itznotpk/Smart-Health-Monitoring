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
        input { margin: 8px; padding: 8px; }
        button { padding: 10px 20px; margin-top: 15px; }
        .result { font-size: 22px; margin-top: 20px; font-weight: bold; }
        .explanation { font-size: 16px; margin-top: 10px; }
        .green { color: green; }
        .red { color: red; }
        .orange { color: orange; }
        table { margin: 20px auto; border-collapse: collapse; width: 80%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .guideline-box { border: 1px solid #ddd; padding: 10px; margin: 10px auto; width: 80%; }
    </style>
</head>
<body>
    <h1>AI-Powered Diabetes Checker</h1>
    <div class="container">
        <p><strong>Disclaimer:</strong> This tool is for informational purposes only and uses Malaysian clinical guidelines. Consult a doctor for a proper diagnosis.</p>
        <form method="post" enctype="multipart/form-data" onsubmit="showLoading()">
            <label for="file">Upload Clinical Report (PDF):</label><br>
            <input type="file" name="file" accept=".pdf" required><br>
            <button type="submit">Analyze Report</button>
        </form>
        <button onclick="location.reload();">Refresh and Reupload</button>

        {% if diagnosis %}
        <div class="result {{ overall_color }}">
            {{ overall_result }}
        </div>
        <table>
            <tr><th>Metric</th><th>Category</th><th>Value</th><th>Unit</th><th>Status</th></tr>
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
                    # Determine overall result based on worst status
                    statuses = [d['status'] for d in diagnosis.values()]
                    if any('Diabetes' in s for s in statuses):
                        overall_result = 'Diabetes Indicated'
                        overall_color = 'red'
                    elif any('Prediabetes' in s for s in statuses):
                        overall_result = 'Prediabetes Indicated'
                        overall_color = 'orange'
                    else:
                        overall_result = 'Normal'
                        overall_color = 'green'
                else:
                    error = "No diabetes-related information found in the report."

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

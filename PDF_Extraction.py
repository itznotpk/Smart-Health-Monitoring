from flask import Flask, render_template_string, request
import re
import pdfplumber
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
    </style>
</head>
<body>
    <h1>AI-Powered Diabetes Checker</h1>
    <div class="container">
        <form method="post" enctype="multipart/form-data">
            <label for="file">Upload Clinical Report (PDF):</label><br>
            <input type="file" name="file" accept=".pdf" required><br>
            <button type="submit">Analyze Report</button>
        </form>

        {% if result %}
        <div class="result {{ color }}">
            {{ result }}
        </div>
        <div class="explanation">
            {{ explanation }}
        </div>
        {% endif %}

        {% if error %}
        <div class="result red">
            {{ error }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# ==========================
# Routes
# ==========================
@app.route("/", methods=["GET", "POST"])
def home():
    result, color, explanation, error = None, None, None, None

    if request.method == "POST":
        if 'file' not in request.files:
            error = "No file part"
            return render_template_string(html_page, error=error)

        file = request.files['file']
        if file.filename == '':
            error = "No selected file"
            return render_template_string(html_page, error=error)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                # Extract text from PDF
                text = ''
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + '\n'

                # Parse for blood glucose level (assuming pattern like "Glucose: 120 mg/dL" or similar)
                glucose_match = re.search(r'(?:glucose|blood sugar|fasting blood glucose|fbs)\s*[:=]?\s*(\d+\.?\d*)\s*(?:mg/dl|mg/dL)?', text, re.IGNORECASE)
                if glucose_match:
                    glucose = float(glucose_match.group(1))

                    # Determine range (assuming fasting blood glucose)
                    if glucose < 70:
                        result = "Low Blood Sugar (Hypoglycemia)"
                        color = "red"
                        explanation = f"Glucose level: {glucose} mg/dL. This is below the safe range. Seek medical attention."
                    elif 70 <= glucose <= 99:
                        result = "Normal"
                        color = "green"
                        explanation = f"Glucose level: {glucose} mg/dL. Your blood sugar is in the safe range."
                    elif 100 <= glucose <= 125:
                        result = "Prediabetes"
                        color = "orange"
                        explanation = f"Glucose level: {glucose} mg/dL. This indicates prediabetes. Consult a doctor."
                    else:
                        result = "Diabetes"
                        color = "red"
                        explanation = f"Glucose level: {glucose} mg/dL. This indicates diabetes. Seek medical advice."
                else:
                    error = "Could not find blood glucose information in the report."

                # Optionally, look for HbA1c
                hba1c_match = re.search(r'(?:hba1c|a1c)\s*[:=]?\s*(\d+\.?\d*)\s*(?:%|\%)?', text, re.IGNORECASE)
                if hba1c_match:
                    hba1c = float(hba1c_match.group(1))
                    if explanation:
                        explanation += f" | HbA1c: {hba1c}%"
                    else:
                        if hba1c < 5.7:
                            result = "Normal"
                            color = "green"
                            explanation = f"HbA1c: {hba1c}%. This is in the normal range."
                        elif 5.7 <= hba1c < 6.5:
                            result = "Prediabetes"
                            color = "orange"
                            explanation = f"HbA1c: {hba1c}%. This indicates prediabetes."
                        else:
                            result = "Diabetes"
                            color = "red"
                            explanation = f"HbA1c: {hba1c}%. This indicates diabetes."

                if not result and not error:
                    error = "No diabetes-related information found in the report."

            except Exception as e:
                error = f"Error processing PDF: {str(e)}"
            finally:
                # Clean up the uploaded file
                os.remove(file_path)
        else:
            error = "Invalid file type. Please upload a PDF."

    return render_template_string(html_page, result=result, color=color, explanation=explanation, error=error)

if __name__ == "__main__":
    app.run(debug=True)

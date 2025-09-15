from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import requests
import json


def query_ollama_api(prompt, model="llama3"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True   # tells Ollama to stream back tokens
    }

    print ("\n[Ollama] Sending request to Ollama API...")

    with requests.post(url, 
        json=payload, stream=True) as r:
        output = ""
        for line in r.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    chunk = data["response"]
                    print(chunk, end="", flush=True)  # print live like CMD
                    output += chunk
                if data.get("done", False):
                    break
        return output.strip()


# ==========================
# Flask App
# ==========================
app = Flask(__name__)

# ==========================
# Load Dataset & Train Model
# ==========================
df = pd.read_csv("health_multiclass_dataset.csv")

X = df[["HeartRate", "SpO2", "BloodPressure", "Temperature"]].values
y = df["Label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(4, activation="softmax")
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=50, batch_size=16,
          callbacks=[early_stop],
          verbose=0)

# ==========================
# HTML Page
# ==========================
html_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Universiti Malaya Health Monitor Kiosk</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; background-color: #f8f9fa; }
        .container { margin-top: 30px; }
        input { margin: 8px; padding: 8px; width: 200px; }
        button { padding: 10px 20px; margin-top: 15px; }
        .result { font-size: 22px; margin-top: 20px; font-weight: bold; }
        .explanation { font-size: 16px; margin-top: 10px; }
        .recommendation { font-size: 16px; margin-top: 20px; color: blue; white-space: pre-line; }
        .green { color: green; }
        .red { color: red; }
        .orange { color: orange; }
        .purple { color: purple; }
    </style>
</head>
<body>
    <h1>AI-Powered Community Health Kiosk</h1>
    <div class="container">
        <form method="post">
            <input type="number" step="any" name="Age" placeholder="Age (years)" required><br>
            <input type="number" step="any" name="HeartRate" placeholder="Heart Rate (bpm)" required><br>
            <input type="number" step="any" name="SpO2" placeholder="SpO₂ (%)" required><br>
            <input type="number" step="any" name="BloodPressure" placeholder="Blood Pressure (mmHg)" required><br>
            <input type="number" step="any" name="Temperature" placeholder="Body Temp (°C)" required><br>
            <button type="submit">Check Health</button>
        </form>

        {% if result %}
        <div class="result {{ color }}">
            {{ result }}
        </div>
        <div class="explanation">
            {{ explanation }}
        </div>
        <div class="recommendation">
            <strong>AI Recommendation:</strong><br>
            {{ recommendation }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# ==========================
# LLM Integration (Ollama)
# ==========================
def generate_recommendation(age, hr, spo2, bp, temp, diagnosis, confidence):
    prompt = f"""
    Patient Information:
    - Age: {age}
    - Heart Rate: {hr} bpm
    - SpO₂: {spo2} %
    - Blood Pressure: {bp} mmHg
    - Temperature: {temp} °C

    AI Health Model Prediction: {diagnosis} (Confidence: {confidence:.1f}%)

    Based on this, give short and practical recommendations:
    - Immediate advice
    - Lifestyle suggestion
    - Whether medical attention is needed
    """
    response = query_ollama_api(prompt, model="llama3:latest")
    print("[DEBUG] Ollama response:", response)  # <-- Add this
    return response


# ==========================
# Routes
# ==========================
@app.route("/", methods=["GET", "POST"])
def home():
    result, color, explanation, recommendation = None, None, None, None
    classes = ["Normal", "Cardiovascular Risk", "Respiratory Issue", "Fever/Infection"]
    colors = ["green", "red", "orange", "purple"]

    if request.method == "POST":
        age = float(request.form["Age"])
        hr = float(request.form["HeartRate"])
        spo2 = float(request.form["SpO2"])
        bp = float(request.form["BloodPressure"])
        temp = float(request.form["Temperature"])

        input_data = np.array([[hr, spo2, bp, temp]])
        if age > 60:
            input_data[0][0] *= 0.9
        elif age < 18:
            input_data[0][0] *= 1.1

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled, verbose=0)[0]
        class_idx = np.argmax(prediction)
        confidence = prediction[class_idx] * 100

        result = classes[class_idx]
        color = colors[class_idx]
        
        reasons = {
            1: "high blood pressure or heart rate",
            2: "low SpO₂ levels",
            3: "elevated temperature",
            0: "all vitals within normal ranges"
        }
        explanation = f"Confidence: {confidence:.1f}%. Flagged due to {reasons.get(class_idx, 'general anomaly')}."

        # 🔹 Generate LLM recommendation
        recommendation = generate_recommendation(age, hr, spo2, bp, temp, result, confidence)

    return render_template_string(
        html_page,
        result=result,
        color=color,
        explanation=explanation,
        recommendation=recommendation
    )

if __name__ == "__main__":
    app.run(debug=True)


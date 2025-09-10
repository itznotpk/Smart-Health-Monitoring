from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

app = Flask(__name__)

# ==========================
# Load & Prepare Dataset
# ==========================
# Note: Ensure health_dataset.csv has columns: HeartRate, SpO2, BloodPressure, Temperature, Label (0=Normal, 1=Cardio, 2=Resp, 3=Fever)
# If not, update dataset to multi-class labels or simulate with synthetic data

df = pd.read_csv("health_multiclass_dataset.csv")

X = df[["HeartRate", "SpO2", "BloodPressure", "Temperature"]].values
y = df["Label"].values  # Multi-class labels: 0, 1, 2, 3

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==========================
# Neural Network (Multi-Class)
# ==========================
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),  # Helps prevent overfitting
    Dense(32, activation="relu"),
    Dense(4, activation="softmax")  # 4 classes: Normal, Cardio, Resp, Fever
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy",  # For multi-class
              metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50, batch_size=16,
                    callbacks=[early_stop],
                    verbose=1)

# ==========================
# Web UI
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
    result, color, explanation = None, None, None
    classes = ["Normal", "Cardiovascular Risk", "Respiratory Issue", "Fever/Infection"]
    colors = ["green", "red", "orange", "purple"]  # Map colors to classes

    if request.method == "POST":
        age = float(request.form["Age"])
        hr = float(request.form["HeartRate"])
        spo2 = float(request.form["SpO2"])
        bp = float(request.form["BloodPressure"])
        temp = float(request.form["Temperature"])

        input_data = np.array([[hr, spo2, bp, temp]])
        # Personalization: Adjust heart rate based on age
        if age > 60:
            input_data[0][0] *= 0.9  # Lower HR baseline for elderly
        elif age < 18:
            input_data[0][0] *= 1.1  # Higher HR baseline for kids

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled, verbose=0)[0]
        class_idx = np.argmax(prediction)
        confidence = prediction[class_idx] * 100

        result = classes[class_idx]
        color = colors[class_idx]
        
        # Simple explanation (expand with SHAP later if needed)
        reasons = {
            1: "high blood pressure or heart rate",
            2: "low SpO2 levels",
            3: "elevated temperature",
            0: "all vitals within normal ranges"
        }
        explanation = f"Confidence: {confidence:.1f}%. Flagged due to {reasons.get(class_idx, 'general anomaly')}."

    return render_template_string(html_page, result=result, color=color, explanation=explanation)


if __name__ == "__main__":
    app.run(debug=True)
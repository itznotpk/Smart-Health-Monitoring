## Dataset

https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

The model is trained on `health_multiclass_dataset.csv`, which should contain the following columns:

- **HeartRate**: Heart rate in beats per minute (bpm).
- **SpO2**: Oxygen saturation percentage (%).
- **BloodPressure**: Blood pressure in mmHg.
- **Temperature**: Body temperature in °C.
- **Label**: Multi-class label (0 = Normal, 1 = Cardiovascular Risk, 2 = Respiratory Issue, 3 = Fever/Infection).

**Note**: The dataset is not included in this repository. You must provide or generate a compatible CSV file with the above structure.

## Requirements

To run this project, install the required Python packages:

```bash
pip install flask pandas numpy scikit-learn tensorflow
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/ai-health-kiosk.git
   cd ai-health-kiosk
   ```
2. Ensure the `health_multiclass_dataset.csv` file is placed in the project root directory.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask application:

   ```bash
   python main.py
   ```
5. Open your browser and navigate to `http://127.0.0.1:5000` to access the health kiosk.

## Usage

1. Enter the following vital signs in the web interface:
   - Age (years)
   - Heart Rate (bpm)
   - SpO2 (%)
   - Blood Pressure (mmHg)
   - Body Temperature (°C)
2. Click "Check Health" to receive a prediction.
3. The result will display the predicted health condition, confidence score, and a brief explanation of the diagnosis.

## Model Details

- **Architecture**: A sequential neural network with:
  - Input layer: 4 features (HeartRate, SpO2, BloodPressure, Temperature).
  - Hidden layers: 64 neurons (ReLU) with 0.3 dropout, followed by 32 neurons (ReLU).
  - Output layer: 4 neurons (softmax) for multi-class classification.
- **Optimizer**: Adam with a learning rate of 0.001.
- **Loss Function**: Sparse categorical crossentropy.
- **Training**: 50 epochs with early stopping (patience=5) to restore the best weights based on validation accuracy.
- **Preprocessing**: Features are scaled using StandardScaler.

## Future Improvements

- Integrate SHAP (SHapley Additive exPlanations) for more detailed prediction explanations.
- Add support for real-time data input from IoT health devices.
- Expand the dataset with more diverse health conditions.
- Enhance the UI with visualizations (e.g., vital sign trends or model confidence charts).

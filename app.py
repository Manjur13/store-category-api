from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and label encoder
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    required_features = [
        'latitude', 'longitude', 'population', 'income', 'youth_ratio',
        'rent', 'crime', 'cafes', 'grocery', 'urban'
    ]

    # Check for missing features
    missing = [f for f in required_features if f not in data]
    if missing:
        return jsonify({'error': f'Missing features: {missing}'}), 400

    # Prepare input as a DataFrame to keep feature names for scaler
    features_df = pd.DataFrame([data], columns=required_features)

    # Scale features
    features_scaled = scaler.transform(features_df)

    # Predict
    prediction = model.predict(features_scaled)

    # Decode label to original category
    predicted_label = label_encoder.inverse_transform(prediction)

    return jsonify({'predicted_category': predicted_label[0]})

if __name__ == '__main__':
    app.run(debug=True)

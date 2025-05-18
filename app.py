from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load your scaler and model (adjust file names if needed)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Make sure all 12 features are present
    required_features = [
        "latitude", "longitude", "population", "income", "youth_ratio",
        "rent", "crime", "cafes", "grocery", "urban", "feature11", "feature12"
    ]

    missing = [f for f in required_features if f not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    # Extract features in correct order
    features = [data[feature] for feature in required_features]

    # Convert to numpy array and reshape for scaler/model
    features_np = np.array(features).reshape(1, -1)

    # Scale features
    features_scaled = scaler.transform(features_np)

    # Predict category
    prediction = model.predict(features_scaled)[0]

    return jsonify({"predicted_category": prediction})

if __name__ == '__main__':
    app.run(debug=True)

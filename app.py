from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model, scaler, and label encoder
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# List of required features (match training data)
required_features = [
    'latitude', 'longitude', 'population_density', 'average_income',
    'age_18_25_ratio', 'rent_per_sqft', 'nearby_cafes', 'nearby_groceries',
    'nearby_electronics', 'nearby_salons', 'crime_rate', 'is_urban'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Check for missing features
    missing = [f for f in required_features if f not in data]
    if missing:
        return jsonify({'error': f'Missing features: {missing}'}), 400

    # Convert to DataFrame with correct column names
    input_df = pd.DataFrame([data], columns=required_features)

    # Scale the input
    scaled_features = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_features)

    # Decode label
    predicted_category = label_encoder.inverse_transform(prediction)[0]

    return jsonify({'predicted_category': predicted_category})

if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[
        data['latitude'],
        data['longitude'],
        data['population'],
        data['income'],
        data['youth_ratio'],
        data['rent'],
        data['crime'],
        data['cafes'],
        data['grocery'],
        data['urban']
    ]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    predicted_label = label_encoder.inverse_transform(prediction)
    return jsonify({'predicted_category': predicted_label[0]})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

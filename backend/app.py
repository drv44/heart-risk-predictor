from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained VotingClassifier (ensemble model)
voting_clf = joblib.load("./models/ensemble.pkl")

@app.route('/')
def home():
    return "API running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)

    # Make prediction using the ensemble model
    prediction = voting_clf.predict(features)
    risk = 'High' if prediction[0] == 1 else 'Low'

    return jsonify({'risk': risk})

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    app.run(host='127.0.0.1', port=5000, debug=True)

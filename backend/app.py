from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Enable CORS if needed for API calls

# Load your pre-trained model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "ensemble.pkl")
voting_clf = joblib.load(model_path)

# Home route to serve the frontend
@app.route('/')
def home():
    return app.send_static_file('index.html')

# API route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    
    # Make prediction using the ensemble model
    prediction = voting_clf.predict(features)
    risk = 'High' if prediction[0] == 1 else 'Low'
    
    return jsonify({'risk': risk})

if __name__ == '__main__':
    # Using host=0.0.0.0 is acceptable for deployment; for local testing, use host='127.0.0.1'
    app.run(host='0.0.0.0', port=5000, debug=True)

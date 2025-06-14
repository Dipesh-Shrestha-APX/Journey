from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load your trained model (make sure model.pkl is in the same folder)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "Selection Predictor API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract input values
        cgpa = data.get('cgpa')
        iq = data.get('iq')

        # Basic validation
        if cgpa is None or iq is None:
            return jsonify({'error': 'Invalid input. Both CGPA and IQ are required.'}), 400

        # Perform prediction
        prediction = model.predict(np.array([[cgpa, iq]]))
        result = 'Selected' if prediction[0] == 1 else 'Not Selected'

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

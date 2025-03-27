import openai
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

# Load trained DQN model
model = load_model("dqn_diabetes_model.h5", compile=False)

# OpenAI API Key (Make sure to keep this secure)
openai.api_key = "your_api"

# Define actions dictionary
actions = {
    0: "Maintain Current Medication & Lifestyle",
    1: "Increase Insulin Dosage",
    2: "Decrease Insulin Dosage",
    3: "Switch to Alternative Medication",
    4: "Suggest Lifestyle Changes (Diet, Exercise)",
    5: "Immediate Doctor Consultation Required"
}

def generate_treatment_explanation(action_text):
    """Generate an AI-based explanation for the treatment."""
    prompt = f"A patient has been recommended to '{action_text}' based on their EHR data. Explain why this is necessary, its benefits, and precautions."
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful medical assistant."},
                  {"role": "user", "content": prompt}]
    )
    
    return response["choices"][0]["message"]["content"]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    patient_data = [
        data['Age'], data['Glucose'], data['HbA1c'], data['Systolic_BP'],
        data['Diastolic_BP'], data['BMI'], data['Medication'], data['Exercise']
    ]

    state = np.array([patient_data]).reshape(1, -1)
    action_id = np.argmax(model.predict(state, verbose=0)[0])
    action_text = actions[action_id]
    explanation = generate_treatment_explanation(action_text)

    return jsonify({"recommendation": action_text, "explanation": explanation})

if __name__ == '__main__':
    app.run(debug=True, port=5500)

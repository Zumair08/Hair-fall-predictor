from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model, encoders = joblib.load('model/hair_loss_model.pkl')

# Recommendation data
RECOMMENDATIONS = {
    'diet': [
        "Increase protein intake (eggs, fish, nuts)",
        "Eat iron-rich foods (spinach, lentils, red meat)",
        "Consume omega-3 fatty acids (salmon, walnuts)",
        "Add biotin-rich foods (sweet potatoes, almonds)"
    ],
    'haircare': [
        "Use mild, sulfate-free shampoo",
        "Massage scalp with coconut/olive oil 2-3 times weekly",
        "Avoid excessive heat styling",
        "Trim hair every 6-8 weeks"
    ],
    'stress': [
        "Practice meditation 10 minutes daily",
        "Get 7-8 hours of sleep",
        "Try yoga or deep breathing exercises",
        "Take regular breaks during work"
    ]
}

@app.route('/')
def home():
    return render_template('home.html')  # Your new homepage

@app.route('/predictor')
def predictor():
    return render_template('index.html')  # Your existing prediction page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive JSON data from frontend
        data = request.json
        input_df = pd.DataFrame([data])

        # Apply label encoders
        for col in input_df.columns:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col].astype(str))
            else:
                return jsonify({'error': f'Unexpected field: {col}'}), 400

        # Predict
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]

        # Build recommendations
        recommendations = {
            'general': "Maintain a healthy lifestyle with balanced diet and proper hair care.",
            'specific': {category: RECOMMENDATIONS[category] for category in RECOMMENDATIONS}
        }

        if prediction[0] == 1:
            recommendations['general'] = "Your results indicate a higher risk of hair fall. Please consult a dermatologist."

        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'recommendations': recommendations,
            'message': 'Success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

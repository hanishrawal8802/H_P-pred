from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('house_price_model.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/')
def home():
    print("🏠 Home route accessed")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("📊 Prediction request received")
        # Get data from form
        size_sqft = float(request.form['size_sqft'])
        bedrooms = int(request.form['bedrooms'])
        house_age = int(request.form['house_age'])
        
        print(f"Features - Size: {size_sqft}, Bedrooms: {bedrooms}, Age: {house_age}")
        
        # Prepare features for prediction
        features = np.array([[size_sqft, bedrooms, house_age]])
        
        # Make prediction
        prediction = model.predict(features)
        
        print(f"Prediction: ${prediction[0]:,.2f}")
        
        # Return result
        return render_template('index.html', 
                             prediction_text=f'Predicted House Price: ${prediction[0]:,.2f}')
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}')

@app.route('/health')
def health():
    return "✅ Server is running!"

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("🌐 Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, host='127.0.0.1', port=5000)
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import os
from waitress import serve

app = Flask(__name__)
# CORS(app, origins=["http://localhost:3000"])
CORS(app)

class HeartDiseaseModel:
    def __init__(self):
        self.model = None
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        self.load_or_train_model()

    def generate_training_data(self):
        """Generate synthetic training data"""
        np.random.seed(42)
        n_samples = 1000

        data = {
            'age': np.clip(np.random.normal(55, 15, n_samples), 20, 100).astype(int),
            'sex': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'cp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
            'trestbps': np.clip(np.random.normal(130, 20, n_samples), 90, 200).astype(int),
            'chol': np.clip(np.random.normal(240, 50, n_samples), 100, 600).astype(int),
            'fbs': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'restecg': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.4, 0.1]),
            'thalach': np.clip(np.random.normal(150, 25, n_samples), 60, 220).astype(int),
            'exang': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'oldpeak': np.clip(np.random.exponential(1, n_samples), 0, 6.2),
            'slope': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2]),
            'ca': np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
            'thal': np.random.choice([1, 2, 3], n_samples, p=[0.1, 0.7, 0.2]),
        }
        df = pd.DataFrame(data)
         #  realistic target variable
        risk_score = (
            df['age'] * 0.08 +
            df['sex'] * 0.25 +
            df['cp'] * 0.35 +
            (df['trestbps'] - 120) * 0.015 +
            (df['chol'] - 200) * 0.008 +
            df['fbs'] * 0.15 +
            df['exang'] * 0.45 +
            df['oldpeak'] * 0.7 +
            df['ca'] * 0.5
        )
        probability = 1 / (1 + np.exp(-(risk_score - risk_score.mean()) / risk_score.std()))   
        df['target'] = (probability + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)

        return df
    def load_or_train_model(self):
        """Load existing model or train a new one"""
        model_path = 'heart_disease_model.pkl'  

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print("Model loaded from disk")

        else:
             print("🔄 Training new model...")
             self.train_model()
             joblib.dump(self.model, model_path)
             print(" Model trained and saved") 

    def train_model(self):  
        """Train the Random Forest model"""
        df = self.generate_training_data()

        X = df[self.feature_names]
        y = df['target']

        self.model = RandomForestClassifier( 
                 n_estimators=100,
                 max_depth=10,
                 min_samples_split=5,
                 min_samples_leaf=2,
                 random_state=42
             )
        self.model.fit(X, y)

             # Print training accuracy
        train_accuracy = self.model.score(X, y)
        print(f"📊 Training Accuracy: {train_accuracy:.3f}")

    def predict(self, input_data):
        """Make prediction on input data"""
        try: 
             # Convert input to DataFrame
            input_df = pd.DataFrame([input_data], columns=self.feature_names)   
             # Make prediction
            probability = self.model.predict_proba(input_df)[0, 1]
            prediction = self.model.predict(input_df)[0] 

             # Get feature importance
            feature_importance = dict(zip(
                 self.feature_names, 
                 self.model.feature_importances_
                ))
            # Generate risk explanation
            explanation = self.generate_explanation(input_data, probability)

            return {
                    'probability': float(probability),
                    'prediction': int(prediction),
                    'risk_level': self.get_risk_level(probability),
                    'feature_importance': feature_importance,
                    'explanation': explanation,
                    'success': True
                 }
        except Exception as e:
             return {
                'success': False,
                'error': str(e)
            }

    def get_risk_level(self, probability):   
        """Determine risk level based on probability"""
        if probability < 0.3:
             return 'Low'
        elif probability < 0.6:
                  return 'Medium'
        else:
                 return 'High' 
             
    def generate_explanation(self, input_data, probability):
        """Generate human-readable explanation of the prediction"""
        factors = []   

        if input_data['age'] > 55:
                  factors.append(f"Age ({input_data['age']} years)") 
        if input_data['sex'] == 1:  # Male
                 factors.append("Male gender")  
        if input_data['cp'] in [1, 2, 3]:  # Any chest pain
                 pain_types = ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"]       
                 factors.append(f"Chest pain type: {pain_types[input_data['cp']]}")
        if input_data['trestbps'] > 140:
                 factors.append(f"High resting blood pressure ({input_data['trestbps']} mm Hg)")   
        if input_data['chol'] > 240:
                 factors.append(f"High cholesterol ({input_data['chol']} mg/dL)")    

        if input_data['fbs'] == 1:
                factors.append("High fasting blood sugar")
        if input_data['exang'] == 1:
                 factors.append("Exercise-induced angina") 
        if input_data['oldpeak'] > 1.0:
                 factors.append(f"ST depression ({input_data['oldpeak']})") 

        if input_data['ca'] > 0:
                 factors.append(f"Major vessels colored: {input_data['ca']}")  

        explanation = f"Based on your health profile, the model predicts a {probability:.1%} risk of heart disease. "

        if factors:
                 explanation += "Key contributing factors include: " + ", ".join(factors[:3]) + ". "   

        if probability > 0.6:
                 explanation += "We recommend consulting a healthcare professional for further evaluation."
        elif probability > 0.3:
                 explanation += "Consider lifestyle modifications and regular health check-ups."  
        else:    
                  explanation += "Maintain your current healthy lifestyle with regular exercise and balanced diet."

                  return explanation
            # Initialize model
model = HeartDiseaseModel()

@app.route('/')
def home():
    return jsonify({
         "message": "Heart Disease Prediction API",
         "status": "active",
         "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)"
         }
    }) 
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model.model is not None})
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

         # Validate required fields
        required_fields = model.feature_names
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }), 400
    # Make prediction
        result = model.predict(data)
        
        if result['success']:
            return jsonify(result)
        else:
             return jsonify({
                "success": False,
                "error": result['error']
            }), 500
    except Exception as e:
           return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }), 500

@app.route('/features')
def get_features():
     """Return feature descriptions for the UI"""
     feature_descriptions = {
        'age': 'Age in years',
        'sex': 'Sex (0: Female, 1: Male)',
        'cp': 'Chest pain type (0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic)',
        'trestbps': 'Resting blood pressure (mm Hg)',
        'chol': 'Cholesterol (mg/dL)',
        'fbs': 'Fasting blood sugar > 120 mg/dL (0: False, 1: True)',
        'restecg': 'Resting electrocardiographic results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (0: No, 1: Yes)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of the peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping)',
        'ca': 'Number of major vessels (0-3) colored by fluoroscopy',  
        'thal': 'Thalassemia (1: Normal, 2: Fixed defect, 3: Reversible defect)'
    }
     return jsonify(feature_descriptions)

if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=5000) 
    import os
    if os.name == 'nt':  # Windows
        try:
          from waitress import serve
          print("🔹 Running on Waitress (Windows production)")
          print("🔹 Server running at: http://0.0.0.0:5000")
          print("🔹 Press Ctrl+C to stop the server")
          serve(app, host='0.0.0.0', port=5000)
        except ImportError:
              print("⚠️  Waitress not installed, falling back to Flask development server")
              print("💡 Install waitress: pip install waitress")
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        # use Gunicorn from terminal:
        #  print("🔹 On Linux/macOS, run with Gunicorn: gunicorn -w 4 -b 0.0.0.0:5000 app:app")
        #  app.run(debug=False, host='0.0.0.0', port=5000)  
        # Check if running in production environment
        if os.environ.get('PRODUCTION'):
             print("🔹 Production mode - should use Gunicorn directly")
             print("💡 Run with: gunicorn -w 4 -b 0.0.0.0:5000 app:app") 

            # In production, this shouldn't be reached when using gunicorn directly
             from gunicorn.app.base import BaseApplication   
             # Fallback to Flask server if gunicorn isn't available
             app.run(debug=False, host='0.0.0.0', port=5000) 
        else:
              # Development mode
             print("🔹 Development mode - using Flask server")
             print("🔹 For production, run with: gunicorn -w 4 -b 0.0.0.0:5000 app:app") 
             app.run(debug=True, host='0.0.0.0', port=5000)        
         
 
        
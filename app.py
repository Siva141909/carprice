from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the pickle files
Gradient_Boosting_model = pickle.load(open('Gradient_Boosting.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# Define feature order
FEATURE_ORDER = ['model', 'vehicle_age', 'km_driven', 'seller_type', 'fuel_type',
                 'transmission_type', 'mileage', 'engine', 'max_power', 'seats']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form
        
        # Map form data to features
        model = form_data['model']
        vehicle_age = int(form_data['vehicle_age'])
        km_driven = int(form_data['km_driven'])
        seller_type = form_data['seller_type']
        fuel_type = form_data['fuel_type']
        transmission_type = form_data['transmission_type']
        mileage = float(form_data['mileage'])
        engine = int(form_data['engine'])
        max_power = float(form_data['max_power'])
        seats = int(form_data['seats'])

        # Prepare data for prediction
        input_data = pd.DataFrame({
            'model': [model],
            'vehicle_age': [vehicle_age],
            'km_driven': [km_driven],
            'seller_type': [seller_type],
            'fuel_type': [fuel_type],
            'transmission_type': [transmission_type],
            'mileage': [mileage],
            'engine': [engine],
            'max_power': [max_power],
            'seats': [seats],
        })

        # Encode 'model' using LabelEncoder
        input_data['model'] = label_encoder.transform(input_data['model'])

        # Preprocess the input data
        processed_data = preprocessor.transform(input_data)

        # Predict the price
        predicted_price = Gradient_Boosting_model.predict(processed_data)

        # Return the prediction
        return render_template('index.html', prediction_text=f"Predicted Price: ₹{predicted_price[0]:,.2f}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

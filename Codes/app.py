from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Directory paths
MODEL_PATH = '../Pickle/'
TEMPLATE_PATH = './templates/'

# Create templates directory if it doesn't exist
os.makedirs(TEMPLATE_PATH, exist_ok=True)

# Try to load the best model available
def load_best_model():
    model_preference = [
        'CatBoost_Advanced.pkl',  # First try advanced model
        'CatBoost',                # Then try original model
        'Extra Tree_Advanced.pkl', # Then next best advanced model
        'Extra Tree',              # Then original version
        'Random Forest_Advanced.pkl',
        'Random Forest'
    ]
    
    for model_name in model_preference:
        try:
            model_path = os.path.join(MODEL_PATH, model_name)
            model = pickle.load(open(model_path, 'rb'))
            print(f"Loaded model: {model_name}")
            return model, model_name
        except:
            continue
    
    print("No models found. Please train models first.")
    return None, None

# Load model at startup
model, model_name = load_best_model()

# Function to make prediction
def predict_price(data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame(data, index=[0])
    
    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)
        return prediction
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Get input data from the form
        data = {
            'Year': int(request.form['year_of_purchase']),
            'Years_used': 2023 - int(request.form['year_of_purchase']),
            'Name': request.form['manufacturer'],
            'Fuel_Type': request.form['fuel_type'],
            'Kilometers_Driven': float(request.form['kilometers_driven']),
            'Engine': float(request.form['engine']),
            'Power': float(request.form['power']),
            'Seats': float(request.form['seats']),
            'Transmission': request.form['transmission'],
            'Owner_Type': request.form['owner_type']
        }
        
        # Make prediction
        prediction = predict_price(data)
    
    return render_template('index.html', prediction=prediction, model_name=model_name)

# Create index.html template
INDEX_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Used Car Price Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f5f5f5;
            padding: 20px;
        }
        .container {
            background-color: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .prediction-result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 5px;
        }
        h1 {
            color: #343a40;
            margin-bottom: 30px;
        }
        label {
            font-weight: 500;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center">Used Car Price Prediction</h1>
        
        <form method="post" class="row g-3">
            <div class="col-md-6">
                <label for="manufacturer" class="form-label">Manufacturer</label>
                <select name="manufacturer" class="form-select" required>
                    <option value="Maruti">Maruti</option>
                    <option value="Hyundai">Hyundai</option>
                    <option value="Honda">Honda</option>
                    <option value="Audi">Audi</option>
                    <option value="Toyota">Toyota</option>
                    <option value="Mahindra">Mahindra</option>
                    <option value="Tata">Tata</option>
                    <option value="Ford">Ford</option>
                    <option value="Chevrolet">Chevrolet</option>
                    <option value="Renault">Renault</option>
                </select>
            </div>
            
            <div class="col-md-6">
                <label for="year_of_purchase" class="form-label">Year of Purchase</label>
                <input type="number" class="form-control" name="year_of_purchase" min="1990" max="2023" value="2015" required>
            </div>
            
            <div class="col-md-6">
                <label for="fuel_type" class="form-label">Fuel Type</label>
                <select name="fuel_type" class="form-select" required>
                    <option value="Petrol">Petrol</option>
                    <option value="Diesel">Diesel</option>
                    <option value="CNG">CNG</option>
                    <option value="LPG">LPG</option>
                    <option value="Electric">Electric</option>
                </select>
            </div>
            
            <div class="col-md-6">
                <label for="transmission" class="form-label">Transmission</label>
                <select name="transmission" class="form-select" required>
                    <option value="Manual">Manual</option>
                    <option value="Automatic">Automatic</option>
                </select>
            </div>
            
            <div class="col-md-6">
                <label for="owner_type" class="form-label">Owner Type</label>
                <select name="owner_type" class="form-select" required>
                    <option value="First">First Owner</option>
                    <option value="Second">Second Owner</option>
                    <option value="Third">Third Owner</option>
                    <option value="Fourth & Above">Fourth & Above</option>
                </select>
            </div>
            
            <div class="col-md-6">
                <label for="kilometers_driven" class="form-label">Kilometers Driven</label>
                <input type="number" class="form-control" name="kilometers_driven" min="0" max="500000" value="50000" required>
            </div>
            
            <div class="col-md-4">
                <label for="engine" class="form-label">Engine (CC)</label>
                <input type="number" class="form-control" name="engine" min="500" max="5000" value="1200" required>
            </div>
            
            <div class="col-md-4">
                <label for="power" class="form-label">Power (BHP)</label>
                <input type="number" class="form-control" name="power" min="30" max="500" value="80" required>
            </div>
            
            <div class="col-md-4">
                <label for="seats" class="form-label">Number of Seats</label>
                <input type="number" class="form-control" name="seats" min="2" max="10" value="5" required>
            </div>
            
            <div class="col-12 text-center mt-4">
                <button type="submit" class="btn btn-primary">Predict Price</button>
            </div>
        </form>
        
        {% if prediction %}
        <div class="prediction-result bg-light">
            <h3 class="text-center">Prediction Result</h3>
            <p class="lead text-center">The estimated price of the car is <strong>₹ {{ prediction }} Lakhs</strong></p>
            <p class="text-center text-muted">Model used: {{ model_name }}</p>
        </div>
        {% endif %}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

# Write the template to a file
with open(os.path.join(TEMPLATE_PATH, 'index.html'), 'w') as f:
    f.write(INDEX_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from data_processor import SalesForecaster
import os
import datetime
import json
from pymongo import MongoClient
from bson.objectid import ObjectId

app = Flask(__name__)
CORS(app)

# Configure JWT
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'super-secret-key-for-development')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(days=1)
jwt = JWTManager(app)

# MongoDB connection
mongo_uri = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
client = MongoClient(mongo_uri)
db = client['inventory_forecast']
users_collection = db['users']
centers_collection = db['distribution_centers']
products_collection = db['products']

# Load forecasting model
model_directory = os.environ.get('MODEL_DIR', './model')
forecaster = SalesForecaster.load(model_directory)

# Helper to convert MongoDB ObjectId to string
def json_serialize(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    raise TypeError("Type not serializable")

# User authentication routes
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({"message": "Missing email or password"}), 400
    
    existing_user = users_collection.find_one({'email': data['email']})
    if existing_user:
        return jsonify({"message": "User already exists"}), 409
    
    hashed_password = generate_password_hash(data['password'])
    
    new_user = {
        'email': data['email'],
        'password': hashed_password,
        'company_name': data.get('company_name', ''),
        'role': data.get('role', 'user'),
        'created_at': datetime.datetime.utcnow()
    }
    
    users_collection.insert_one(new_user)
    
    return jsonify({"message": "User registered successfully"}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({"message": "Missing email or password"}), 400
    
    user = users_collection.find_one({'email': data['email']})
    
    if not user or not check_password_hash(user['password'], data['password']):
        return jsonify({"message": "Invalid credentials"}), 401
    
    access_token = create_access_token(identity=str(user['_id']))
    
    return jsonify({
        "message": "Login successful",
        "token": access_token,
        "user": {
            "email": user['email'],
            "company_name": user.get('company_name', ''),
            "role": user.get('role', 'user')
        }
    }), 200

# Distribution center routes
@app.route('/api/centers', methods=['GET'])
@jwt_required()
def get_centers():
    centers = list(centers_collection.find({}))
    return jsonify({
        "centers": json.loads(json.dumps(centers, default=json_serialize))
    }), 200

@app.route('/api/centers/<center_id>', methods=['GET'])
@jwt_required()
def get_center(center_id):
    center = centers_collection.find_one({'_id': ObjectId(center_id)})
    if not center:
        return jsonify({"message": "Center not found"}), 404
    
    return jsonify({
        "center": json.loads(json.dumps(center, default=json_serialize))
    }), 200

# Forecasting routes
@app.route('/api/forecast/<center_id>', methods=['GET'])
@jwt_required()
def forecast(center_id):
    try:
        # Get recent sales data for the distribution center
        center = centers_collection.find_one({'_id': ObjectId(center_id)})
        if not center:
            return jsonify({"message": "Center not found"}), 404
        
        # In a real application, you would query your database for recent sales data
        # For this example, we'll use dummy data
        recent_data = pd.DataFrame({
            'sales': np.random.rand(40) * 100 + 50,
            'day_of_week': list(range(7)) * (40 // 7) + list(range(40 % 7)),
            'month': [datetime.datetime.now().month] * 40,
            'inventory_level': np.random.rand(40) * 200 + 100,
        })
        
        # Get product IDs for the center
        product_ids = [p['product_id'] for p in products_collection.find(
            {'center_id': center_id}, {'product_id': 1}
        )]
        
        # If no products found, return empty forecast
        if not product_ids:
            return jsonify({"message": "No products found for this center"}), 404
        
        # Generate forecast
        top_products = forecaster.predict(
            recent_data, 
            top_n=10, 
            product_ids=product_ids[:100]  # Limit to first 100 products
        )
        
        # Convert to list for JSON response
        forecast_list = []
        for _, row in top_products.iterrows():
            product = products_collection.find_one({'product_id': row['product_id']})
            if product:
                forecast_list.append({
                    'product_id': row['product_id'],
                    'product_name': product.get('name', 'Unknown'),
                    'forecast': float(row['forecast']),
                    'category': product.get('category', 'Uncategorized')
                })
        
        return jsonify({
            "center_id": center_id,
            "center_name": center.get('name', 'Unknown'),
            "forecast_date": datetime.datetime.now().isoformat(),
            "forecast_horizon": forecaster.prediction_horizon,
            "top_products": forecast_list
        }), 200
        
    except Exception as e:
        app.logger.error(f"Error generating forecast: {str(e)}")
        return jsonify({"message": "Error generating forecast", "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', 'False') == 'True')

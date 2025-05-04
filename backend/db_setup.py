from pymongo import MongoClient
from werkzeug.security import generate_password_hash
import datetime
import os
import random
import json

# MongoDB connection
mongo_uri = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
client = MongoClient(mongo_uri)
db = client['inventory_forecast']

# Collections
users_collection = db['users']
centers_collection = db['distribution_centers']
products_collection = db['products']
sales_collection = db['sales']

def setup_database():
    """Initialize database with sample data."""
    print("Setting up database with sample data...")
    
    # Clear existing data
    users_collection.delete_many({})
    centers_collection.delete_many({})
    products_collection.delete_many({})
    sales_collection.delete_many({})
    
    # Create admin user
    admin_user = {
        'email': 'admin@example.com',
        'password': generate_password_hash('admin123'),
        'company_name': 'Demo Company',
        'role': 'admin',
        'created_at': datetime.datetime.utcnow()
    }
    users_collection.insert_one(admin_user)
    print("Admin user created")
    
    # Create distribution centers
    distribution_centers = [
        {
            'name': 'Northeast DC',
            'location': {
                'lat': 40.7128,
                'lng': -74.0060
            },
            'address': '123 Main St, New York, NY',
            'capacity': 50000,
            'active': True
        },
        {
            'name': 'Southeast DC',
            'location': {
                'lat': 33.7490,
                'lng': -84.3880
            },
            'address': '456 Peachtree St, Atlanta, GA',
            'capacity': 35000,
            'active': True
        },
        {
            'name': 'Midwest DC',
            'location': {
                'lat': 41.8781,
                'lng': -87.6298
            },
            'address': '789 Michigan Ave, Chicago, IL',
            'capacity': 45000,
            'active': True
        },
        {
            'name': 'Southwest DC',
            'location': {
                'lat': 29.7604,
                'lng': -95.3698
            },
            'address': '321 Texas Blvd, Houston, TX',
            'capacity': 40000,
            'active': True
        },
        {
            'name': 'West Coast DC',
            'location': {
                'lat': 34.0522,
                'lng': -118.2437
            },
            'address': '987 Hollywood Blvd, Los Angeles, CA',
            'capacity': 55000,
            'active': True
        },
        {
            'name': 'Pacific Northwest DC',
            'location': {
                'lat': 47.6062,
                'lng': -122.3321
            },
            'address': '654 Pine St, Seattle, WA',
            'capacity': 30000,
            'active': True
        }
    ]
    
    center_results = centers_collection.insert_many(distribution_centers)
    center_ids = center_results.inserted_ids
    print(f"Created {len(center_ids)} distribution centers")
    
    # Create connections between distribution centers
    connections = [
        {'from': 0, 'to': 1},  # Northeast to Southeast
        {'from': 0, 'to': 2},  # Northeast to Midwest
        {'from': 1, 'to': 3},  # Southeast to Southwest
        {'from': 2, 'to': 3},  # Midwest to Southwest
        {'from': 3, 'to': 4},  # Southwest to West Coast
        {'from': 4, 'to': 5},  # West Coast to Pacific Northwest
        {'from': 5, 'to': 2},  # Pacific Northwest to Midwest
    ]
    
    for conn in connections:
        from_center = centers_collection.find_one({'_id': center_ids[conn['from']]})
        to_center = centers_collection.find_one({'_id': center_ids[conn['to']]})
        
        # Add connection to 'from' center
        centers_collection.update_one(
            {'_id': center_ids[conn['from']]},
            {'$push': {'connections': str(center_ids[conn['to']])}}
        )
        
        # Add connection to 'to' center (bidirectional)
        centers_collection.update_one(
            {'_id': center_ids[conn['to']]},
            {'$push': {'connections': str(center_ids[conn['from']])}}
        )
    
    print("Created connections between distribution centers")
    
    # Create product categories
    categories = [
        'Electronics', 'Home Goods', 'Clothing', 'Sports Equipment', 
        'Toys', 'Food', 'Health & Beauty', 'Office Supplies'
    ]
    
    # Create products for each center
    products = []
    for i, center_id in enumerate(center_ids):
        for j in range(1, 31):  # 30 products per center
            category = random.choice(categories)
            product = {
                'product_id': f"P{i+1}-{j:03d}",
                'name': f"{category} Product {j}",
                'center_id': str(center_id),
                'category': category,
                'price': round(random.uniform(5.99, 199.99), 2),
                'weight': round(random.uniform(0.1, 20.0), 1),
                'inventory': random.randint(10, 200)
            }
            products.append(product)
    
    products_collection.insert_many(products)
    print(f"Created {len(products)} products")
    
    # Create sample sales data
    print("Sample database setup complete")

if __name__ == "__main__":
    setup_database()

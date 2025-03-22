from flask import Flask, render_template, request, jsonify
import pickle
import model  # Importing model.py

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    username = request.form['username']
    
    # Check if user exists
    user_exists = model.check_user_exists(username)  
    if not user_exists:
        return jsonify({"error": "User not found. Please enter a valid username."})

    # Get recommendations
    recommendations = model.get_top_5_products(username)
    
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)

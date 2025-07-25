from flask import Flask, request, jsonify
import mysql.connector
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# MySQL database connection details
host = "srv1814.hstgr.io"
user = "u352881525_mapt"
password = "Chathu6@ac"
database = "u352881525_mapt_web"

# Function to get data from the MySQL database
def get_destinations_from_db():
    # Connect to the database
    connection = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )

    cursor = connection.cursor()

    # Query to fetch the destination data
    cursor.execute("SELECT * FROM destinations")

    # Fetch all the rows
    rows = cursor.fetchall()

    # Column names corresponding to the database table
    columns = ['id', 'name', 'category', 'location', 'description', 'food_and_drink', 'culture_and_heritage', 
               'nature_and_adventure', 'art_and_creativity', 'wellness_and_relaxation', 'sustainable_travel', 
               'urban_exploration', 'community_and_social_experiences']

    # Convert the data into a pandas DataFrame
    destinations = pd.DataFrame(rows, columns=columns)

    # Close the database connection
    cursor.close()
    connection.close()

    return destinations

# Function to calculate match percentage using SVM
def calculate_match_percentage(user_preferences, destinations):
    # Features (destination features)
    features = ['food_and_drink', 'culture_and_heritage', 'nature_and_adventure', 
                'art_and_creativity', 'wellness_and_relaxation', 'sustainable_travel', 
                'urban_exploration', 'community_and_social_experiences']
    
    # Prepare the data
    X = destinations[features]
    
    # Calculate dynamic match scores based on user preferences
    match_scores = []
    for _, row in X.iterrows():
        # Calculate the absolute difference between user preferences and destination features
        differences = [abs(user_preferences[feature] - row[feature]) for feature in features]
        
        # Higher match score if differences are lower
        # The score is normalized to range from 1 (worst match) to 10 (best match)
        match_score = 10 - (sum(differences) / len(features))  # Lower difference means higher score
        match_scores.append(match_score)
    
    # Prepare target variable y as dynamic match scores
    y = match_scores
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardizing the features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize the SVM model
    svm_model = SVR(kernel='linear')  # Linear kernel for simplicity
    
    # Train the model
    svm_model.fit(X_train_scaled, y_train)
    
    # Predict match score for the user's preferences
    preferences = pd.DataFrame(user_preferences, index=[0])
    preferences_scaled = scaler.transform(preferences)
    predicted_match = svm_model.predict(preferences_scaled)
    
    # Convert predicted match score to percentage (out of 100)
    predicted_match_percentage = predicted_match[0] * 10  # Scale 1-10 to 0-100
    
    # Calculate match scores for all destinations based on user preferences
    destinations['match_score'] = svm_model.predict(scaler.transform(destinations[features])) * 10  # Convert match score to percentage
    
    # Sort the destinations by match score
    recommendations = destinations[['name', 'match_score', 'category', 'location', 'description']]
    recommendations = recommendations.sort_values(by='match_score', ascending=False)

    # Filter destinations with match score greater than 65%
    recommendations = recommendations[recommendations['match_score'] > 65]  # Match score above 65% (out of 100)
    
    return recommendations, predicted_match_percentage

# API endpoint to get recommendations based on user preferences
@app.route('/get_recommendations', methods=['GET'])
def get_recommendations():
    try:
        # Get user preferences from the request parameters
        user_preferences = {
            'food_and_drink': int(request.args.get('food_and_drink')),
            'culture_and_heritage': int(request.args.get('culture_and_heritage')),
            'nature_and_adventure': int(request.args.get('nature_and_adventure')),
            'art_and_creativity': int(request.args.get('art_and_creativity')),
            'wellness_and_relaxation': int(request.args.get('wellness_and_relaxation')),
            'sustainable_travel': int(request.args.get('sustainable_travel')),
            'urban_exploration': int(request.args.get('urban_exploration')),
            'community_and_social_experiences': int(request.args.get('community_and_social_experiences'))
        }

        # Get destinations from MySQL database
        destinations = get_destinations_from_db()

        # Get recommendations and predicted match score
        recommendations, predicted_match = calculate_match_percentage(user_preferences, destinations)

        # Format the recommendations as a list of dictionaries
        recommendation_list = recommendations.to_dict(orient='records')

        # Return the results as JSON
        return jsonify({
            'predicted_match_score': f"{predicted_match:.2f}%",
            'recommendations': recommendation_list
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

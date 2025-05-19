from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic

# Initialize Flask app
app = Flask(__name__)

# Load ML model and encoders
model = pickle.load(open('model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
features = ['Name_x', 'State', 'Type', 'BestTimeToVisit', 'Preferences', 'Gender', 'NumberOfAdults', 'NumberOfChildren']

# Load data
df = pd.read_csv('final_df.csv')
destinations_df = pd.read_csv('Expanded_Destinations.csv')
userhistory_df = pd.read_csv('Final_Updated_Expanded_UserHistory.csv')

# User-item matrix for collaborative filtering
user_item_matrix = userhistory_df.pivot(index='UserID', columns='DestinationID', values='ExperienceRating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)


# Collaborative Filtering
def collaborative_recommend(user_id, user_similarity, user_item_matrix, destinations_df):
    similar_users = user_similarity[user_id - 1]
    similar_users_idx = np.argsort(similar_users)[::-1][1:6]
    similar_user_ratings = user_item_matrix.iloc[similar_users_idx].mean(axis=0)
    recommended_dest_ids = similar_user_ratings.sort_values(ascending=False).head(5).index
    recommendations = destinations_df[destinations_df['DestinationID'].isin(recommended_dest_ids)][[
        'DestinationID', 'Name', 'State', 'Type', 'Popularity', 'BestTimeToVisit']]
    return recommendations


# Popularity prediction
def recommend_destinations(user_input, model, label_encoders, features, data):
    encoded_input = {}
    for feature in features:
        if feature in label_encoders:
            encoded_input[feature] = label_encoders[feature].transform([user_input[feature]])[0]
        else:
            encoded_input[feature] = int(user_input[feature])
    input_df = pd.DataFrame([encoded_input])
    predicted_popularity = model.predict(input_df)[0]
    return predicted_popularity

def run_pathfinding(start, end, algo):
    # Example: simulate straight line for now
    # Later: replace with A* or Dijkstra real logic on graph of cities
    return [start, end]

# Dijkstra’s Algorithm (using geodesic distance)
def dijkstra_shortest_path(start_coord, end_coord, all_coords):
    distances = {coord: float('inf') for coord in all_coords}
    previous = {coord: None for coord in all_coords}
    distances[start_coord] = 0
    unvisited = set(all_coords)

    while unvisited:
        current = min(unvisited, key=lambda coord: distances[coord])
        unvisited.remove(current)

        if current == end_coord:
            break

        for neighbor in all_coords:
            if neighbor == current or neighbor not in unvisited:
                continue
            dist = geodesic(current, neighbor).kilometers
            alt = distances[current] + dist
            if alt < distances[neighbor]:
                distances[neighbor] = alt
                previous[neighbor] = current

    # Reconstruct path
    path = []
    current = end_coord
    while current:
        path.insert(0, current)
        current = previous[current]

    return path, distances[end_coord]


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html',
                           recommended_destinations=pd.DataFrame(),
                           predicted_popularity=None)


@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    user_input = {
        'Name_x': request.form['name'],
        'Type': request.form['type'],
        'State': request.form['state'],
        'BestTimeToVisit': request.form['best_time'],
        'Preferences': request.form['preferences'],
        'Gender': request.form['gender'],
        'NumberOfAdults': request.form['adults'],
        'NumberOfChildren': request.form['children'],
    }

    recommended_destinations = collaborative_recommend(user_id, user_similarity, user_item_matrix, destinations_df)
    predicted_popularity = recommend_destinations(user_input, model, label_encoders, features, df)

    return render_template('recommendation.html',
                           recommended_destinations=recommended_destinations,
                           predicted_popularity=predicted_popularity)


# API for shortest path
@app.route('/shortest_path', methods=['POST'])
def shortest_path():
    data = request.get_json(force=True)
    start = (data['start']['lat'], data['start']['lng'])
    end = (data['end']['lat'], data['end']['lng'])
    algo = data.get('algorithm', 'dijkstra')

    # Call your shortest path algorithm
    path = run_pathfinding(start, end, algo)

    return jsonify({'path': path})  # List of [lat, lng] pairs

    # Run Dijkstra’s algorithm
    if algorithm == 'dijkstra':
        path, distance = dijkstra_shortest_path(start, end, coords)
        return jsonify({
            'path': [{'lat': p[0], 'lng': p[1]} for p in path],
            'distance': distance
        })
    else:
        return jsonify({'error': 'Unsupported algorithm'}), 400


if __name__ == '__main__':
    app.run(debug=True)

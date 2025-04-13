from flask import Flask, request, jsonify, send_from_directory
import random
import torch
import torch.nn as nn
import pandas as pd

app = Flask(__name__, static_folder='static')

# Load your data
df = pd.read_csv("Movie_Ratings_Merged.csv")

user_id_mapping = {id_: idx for idx, id_ in enumerate(df['userId'].unique())}
movie_title_mapping = {title: idx for idx, title in enumerate(df['title'].unique())}

df['user_idx'] = df['userId'].map(user_id_mapping)
df['movie_idx'] = df['title'].map(movie_title_mapping)

popular_titles = df['title'].value_counts().head(200).index.tolist()
movie_db = df[['title']].drop_duplicates().reset_index(drop=True).to_dict(orient='records')
popular_movie_db = [m for m in movie_db if m['title'] in popular_titles and m['title'] in movie_title_mapping]

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, n_factors=50):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, n_factors)
        self.movie_emb = nn.Embedding(num_movies, n_factors)

    def forward(self, user_ids, movie_ids):
        user_vecs = self.user_emb(user_ids)
        movie_vecs = self.movie_emb(movie_ids)
        return (user_vecs * movie_vecs).sum(1)

num_users = len(user_id_mapping)
num_movies = len(movie_title_mapping)

model = MatrixFactorization(num_users, num_movies, n_factors=50)
model.load_state_dict(torch.load('model.pth'))
model.eval()

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/get_movies', methods=['GET'])
def get_movies():
    return jsonify(random.sample(popular_movie_db, 10))

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_ratings = data['ratings']

    user_movie_idxs = [movie_title_mapping[m['title']] for m in user_ratings]
    user_rating_vals = [m['rating'] for m in user_ratings]

    # Use liked movies (rating >= 4) to define taste vector
    liked_movie_idxs = [idx for idx, rating in zip(user_movie_idxs, user_rating_vals) if rating >= 4]

    # If user didn't like anything — fallback to all rated movies
    if not liked_movie_idxs:
        liked_movie_idxs = user_movie_idxs

    liked_movie_tensor = torch.tensor(liked_movie_idxs)

    # Create new user vector by averaging liked movie embeddings
    user_vec = model.movie_emb(liked_movie_tensor).mean(dim=0)

    # Find unseen movies
    rated = set(user_movie_idxs)
    unseen = list(set(movie_title_mapping.values()) - rated)

    movie_tensor = torch.tensor(unseen)
    movie_vecs = model.movie_emb(movie_tensor)

    # Compute similarity scores
    scores = (user_vec * movie_vecs).sum(dim=1)

    top5 = scores.topk(5).indices
    recommended_idxs = [unseen[i] for i in top5]

    idx_to_movie = {idx: title for title, idx in movie_title_mapping.items()}
    recommended_titles = [idx_to_movie[idx] for idx in recommended_idxs]

    return jsonify(recommended_titles)

if __name__ == '__main__':
    app.run(debug=True)

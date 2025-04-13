import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np

# ============================
# 1. Load and Preprocess Data
# ============================

df = pd.read_csv("Movie_Ratings_Merged.csv")

# Map userId and title to unique indices
user_id_mapping = {id_: idx for idx, id_ in enumerate(df['userId'].unique())}
movie_title_mapping = {title: idx for idx, title in enumerate(df['title'].unique())}

df['user_idx'] = df['userId'].map(user_id_mapping)
df['movie_idx'] = df['title'].map(movie_title_mapping)

# Filter out users with fewer than 2 ratings
valid_users = df['userId'].value_counts()
df = df[df['userId'].isin(valid_users[valid_users >= 2].index)]

# Split data into train/test sets (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['userId'], random_state=42)

# ============================
# 2. Create Dataset and Dataloader
# ============================

class MovieRatingDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_idx'].values, dtype=torch.long)
        self.movies = torch.tensor(df['movie_idx'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

train_dataset = MovieRatingDataset(train_df)
test_dataset = MovieRatingDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# ============================
# 3. Define Matrix Factorization Model
# ============================

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, n_factors=50):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, n_factors)
        self.movie_emb = nn.Embedding(num_movies, n_factors)

    def forward(self, user_ids, movie_ids):
        user_vecs = self.user_emb(user_ids)
        movie_vecs = self.movie_emb(movie_ids)
        return (user_vecs * movie_vecs).sum(1)

# Initialize model
num_users = len(user_id_mapping)
num_movies = len(movie_title_mapping)
model = MatrixFactorization(num_users, num_movies, n_factors=50)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================
# 4. Optional Model Training
# ============================

TRAIN_MODEL = False  # Set to True if you want to retrain

if TRAIN_MODEL:
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for user_ids, movie_ids, ratings in train_loader:
            optimizer.zero_grad()
            predictions = model(user_ids, movie_ids)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(ratings)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for user_ids, movie_ids, ratings in test_loader:
                predictions = model(user_ids, movie_ids)
                loss = criterion(predictions, ratings)
                val_loss += loss.item() * len(ratings)

        avg_val_loss = val_loss / len(test_loader.dataset)
        print(f"→ Validation Loss: {avg_val_loss:.4f}")

# ============================
# 5. Evaluation Function
# ============================

user_actual_likes = defaultdict(set)
for row in test_df.itertuples():
    if row.rating >= 4.0:
        user_actual_likes[row.user_idx].add(row.movie_idx)

all_movie_indices = set(df['movie_idx'].unique())

def precision_recall_at_k(model, train_df, test_df, user_actual_likes, k=10):
    model.eval()
    user_pred_scores = defaultdict(list)

    with torch.no_grad():
        for user_idx in user_actual_likes:
            rated_movies = set(train_df[train_df['user_idx'] == user_idx]['movie_idx'])
            unseen_movies = list(all_movie_indices - rated_movies)

            if len(unseen_movies) == 0:
                continue

            user_tensor = torch.tensor([user_idx] * len(unseen_movies), dtype=torch.long)
            movie_tensor = torch.tensor(unseen_movies, dtype=torch.long)

            scores = model(user_tensor, movie_tensor)
            top_k_indices = scores.topk(k).indices
            top_k_movies = [unseen_movies[i] for i in top_k_indices]

            user_pred_scores[user_idx] = top_k_movies

    precisions, recalls = [], []
    for user_idx, recommended in user_pred_scores.items():
        relevant = user_actual_likes[user_idx]
        if not relevant:
            continue
        recommended_set = set(recommended)
        true_positives = len(recommended_set & relevant)
        precision = true_positives / len(recommended)
        recall = true_positives / len(relevant)

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)

    return avg_precision, avg_recall

# ============================
# 6. Evaluate Only When Running Directly
# ============================

if __name__ == "__main__":
    precision, recall = precision_recall_at_k(model, train_df, test_df, user_actual_likes, k=10)
    print(f"Precision@10: {precision:.4f}")
    print(f"Recall@10: {recall:.4f}")

# ============================
# 7. Build Movie Database for Flask
# ============================

def extract_first_genre(genre_list_str):
    try:
        genres = eval(genre_list_str)
        return genres[0] if genres else 'Unknown'
    except:
        return 'Unknown'

df['genre'] = df['genres_parsed'].apply(extract_first_genre)

movie_db = df[['title', 'genre']].drop_duplicates().reset_index(drop=True).to_dict(orient='records')
# Keep only popular movies (top 200 most rated)
# Only include popular movies that exist in movie_title_mapping
popular_titles = df['title'].value_counts().head(200).index.tolist()

popular_movie_db = [movie for movie in movie_db if movie['title'] in popular_titles and movie['title'] in movie_title_mapping]
torch.save(model.state_dict(), 'model.pth')

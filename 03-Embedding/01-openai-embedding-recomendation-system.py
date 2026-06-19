import os
import openai
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

# Load the user_reviews dataset
df = pd.read_csv("99-EN/journey-of-extinction-husun/user_reviews.csv")

# Load the game_descriptions file
with open("99-EN/journey-of-extinction-husun/game_guide.json", "r") as f:
    game_descriptions = json.load(f)

# Define a function to get the embedding vector
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# Get embedding vectors for all games
unique_games = df['game_title'].unique().tolist()
target_game = "Killing God: Hu Sun"  # change the target game name
if target_game not in unique_games:
    unique_games.append(target_game)  # ensure the target game is in the list
game_embeddings = {}
for game in unique_games:
    description = game_descriptions[game]
    game_embeddings[game] = np.array(get_embedding(description))

# Compute each user's embedding vector (the average of the game_description embeddings for all games that user reviewed)
user_vectors = {}
for user_id, group in df.groupby("user_id"):
    user_game_vecs = []
    for idx, row in group.iterrows():
        g_title = row['game_title']
        g_vec = game_embeddings[g_title]
        user_game_vecs.append(g_vec)
    user_vectors[user_id] = np.mean(np.array(user_game_vecs), axis=0)

# Get the embedding vector for "Chronicles of Godslaying: Wukong"
target_vector = game_embeddings[target_game]
# Compute the cosine similarity between each user's embedding vector and the target game's embedding vector
results = []
for user_id, u_vec in user_vectors.items():
    u_vec_reshaped = u_vec.reshape(1, -1)
    t_vec = target_vector.reshape(1, -1)
    similarity = cosine_similarity(u_vec_reshaped, t_vec)[0,0]
    results.append((user_id, similarity))

# Sort and find the users most likely to like "Chronicles of Godslaying: Wukong"
result_df = pd.DataFrame(results, columns=["user_id", f"similarity_to_{target_game}"])
result_df = result_df.sort_values(by=f"similarity_to_{target_game}", ascending=False)
print(f"\nTop 5 users most likely to like {target_game}:")
print(result_df.head())

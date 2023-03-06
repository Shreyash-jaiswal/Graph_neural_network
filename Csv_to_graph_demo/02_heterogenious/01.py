import pandas as pd


##################################################################################################################
#       Loading The DataSet
#       its an anime recomender dataset
#       simple recommendation system can be implemented using GNN 
##################################################################################################################
# !wget -q https://raw.githubusercontent.com/Mayank-Bhatia/Anime-Recommender/master/data/anime.csv
# !wget -q https://raw.githubusercontent.com/Mayank-Bhatia/Anime-Recommender/master/data/rating.csv

anime = pd.read_csv("anime.csv")
rating = pd.read_csv("rating.csv")

print(anime.head())
print(rating.head())

'''
Nodes - Users and Animes (two node types with different features = heterogeneous)
Edges - If a user has rated a movie / the rating (edge weight)
Node Features - The movie attributes and for the users we have no explicit features so we have to figure something out later
Labels - The rating for a movie (link prediction regression task)
'''


##################################################################################################################
#       Extract Node Features
##################################################################################################################

# Sort to define the order of nodes
sorted_df = anime.sort_values(by="anime_id").set_index("anime_id")

# Map IDs to start from 0
sorted_df = sorted_df.reset_index(drop=False)
movie_id_mapping = sorted_df["anime_id"]

# Select node features
node_features = sorted_df[["type", "genre", "episodes"]]
# Convert non-numeric columns
pd.set_option('mode.chained_assignment', None)

# For simplicity I'll just select the first genre here and ignore the others
genres = node_features["genre"].str.split(",", expand=True)
node_features["main_genre"] = genres[0]

# One-hot encoding to movie type
anime_node_features = pd.concat([node_features, pd.get_dummies(node_features["main_genre"])], axis=1, join='inner')
anime_node_features = pd.concat([anime_node_features, pd.get_dummies(anime_node_features["type"])], axis=1, join='inner')
anime_node_features.drop(["genre", "main_genre"], axis=1, inplace=True)
anime_node_features.head(10)


##################################################################################################################
#       Converting to Numpy
##################################################################################################################

# Convert to numpy
x = anime_node_features.to_numpy()
print(x.shape) # [num_movie_nodes x movie_node_feature_dim]


# as we dont have we can insert the dummy or we can calulate something using statistics
# Find out mean rating and number of ratings per user
mean_rating = rating.groupby("user_id")["rating"].mean().rename("mean")
num_rating = rating.groupby("user_id")["rating"].count().rename("count")
user_node_features = pd.concat([mean_rating, num_rating], axis=1)

# Remap user ID (to start at 0)
user_node_features = user_node_features.reset_index(drop=False)
user_id_mapping = user_node_features["user_id"]

# Only keep features 
user_node_features = user_node_features[["mean", "count"]]
user_node_features.head()

# Convert to numpy
x = user_node_features.to_numpy()
x.shape # [num_user_nodes x user_node_feature_dim]

print(rating.head())

# -1 means the user watched but didn't assign a weight
rating["rating"].hist()

# Movies that are part of our rating matrix
rating["anime_id"].unique()

# All movie IDs (e.g. no rating above for 1, 5, 6...)
anime["anime_id"].sort_values().unique()

# We can also see that there are some movies in the rating matrix, for which we have no features (we will drop them here)
print(set(rating["anime_id"].unique()) - set(anime["anime_id"].unique()))
rating = rating[~rating["anime_id"].isin([30913, 30924, 20261])]

# Extract labels
labels = rating["rating"]
print(labels.tail())

# Convert to numpy
y = labels.to_numpy()
print(y.shape)


print("Before remapping...")
print(rating.head())

# Map anime IDs 
movie_map = movie_id_mapping.reset_index().set_index("anime_id").to_dict()
rating["anime_id"] = rating["anime_id"].map(movie_map["index"]).astype(int)
# Map user IDs
user_map = user_id_mapping.reset_index().set_index("user_id").to_dict()
rating["user_id"] = rating["user_id"].map(user_map["index"]).astype(int)

print("After remapping...")
print(rating.head())

edge_index = rating[["user_id", "anime_id"]].values.transpose()
print(edge_index) # [2 x num_edges] 


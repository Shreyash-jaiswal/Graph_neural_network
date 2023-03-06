
##################################################################################################################
#       Loading The DataSet
##################################################################################################################
import pandas as pd


# This is FIFA 21 DataSet present in csv file 
#  Downloading the dataSet 
# !wget -q https://raw.githubusercontent.com/batuhan-demirci/fifa21_dataset/master/data/tbl_player.csv
# !wget -q https://raw.githubusercontent.com/batuhan-demirci/fifa21_dataset/master/data/tbl_player_skill.csv
# !wget -q https://raw.githubusercontent.com/batuhan-demirci/fifa21_dataset/master/data/tbl_team.csv
 

# Load data
player_df = pd.read_csv("tbl_player.csv")   # contains all the player stats
skill_df = pd.read_csv("tbl_player_skill.csv") # contains there skills
team_df = pd.read_csv("tbl_team.csv")   # contains data about there teams


#identification of various properties of graph based on dataset 
    # NODES = here we will represent each player as node
    # Edges = relationship between each player so if they play for same team or not
    # Node Features =  player's position , skills etc
    # Laberls = there overall rating (node-level regression task)

    # Nodes contains usually id and if they dont contain create id as it will be required to know 
    #   if the connection between nodes exists

    # In this graph we can predict the expected overall rating when a player switch from one team to 
    #   the other or new player is observed THEREFORE team assignment is taken as EDGE


# Extract subsets of information
player_df = player_df[["int_player_id", "str_player_name", "str_positions", "int_overall_rating", "int_team_id"]] #elects a subset of columns from a pandas DataFrame called player_df.
skill_df = skill_df[["int_player_id", "int_long_passing", "int_ball_control", "int_dribbling"]]
team_df = team_df[["int_team_id", "str_team_name", "int_overall"]]

# Merging data same concept as primary key and foregign key
# Merging Player_df with Skill_df
player_df = player_df.merge(skill_df, on='int_player_id') # merge method has to be called on one df and passed another df as argument
# Merging team_df with player_df based on team_id
fifa_df = player_df.merge(team_df, on='int_team_id')

# Sort dataframe
fifa_df = fifa_df.sort_values(by="int_overall_rating", ascending=False)
print("Players: ", fifa_df.shape[0])
print(fifa_df.head())

# here the data is combined into a single data frame representing one large (disconnected graph)

# Make sure that we have no duplicate nodes
# print("Printing maximum count per player_id " + str(max(fifa_df["int_player_id"].value_counts())))

# Sort to define the order of nodes
sorted_df = fifa_df.sort_values(by="int_player_id") 
# print(sorted_df) # just arranging based on player_id as main identifier

#################################################################################################################
#       Extracting the Node features
##################################################################################################################

# Select node features
node_features = sorted_df[["str_positions", "int_long_passing", "int_ball_control", "int_dribbling"]]

# Convert non-numeric columns
pd.set_option('mode.chained_assignment', None)
# what is meant by this command is 
"""
    The line pd.set_option('mode.chained_assignment', None) is used to turn off the SettingWithCopyWarning that 
    Pandas may generate when attempting to modify a DataFrame in place through chained indexing.

    Chained indexing refers to when multiple indexing operations are performed in a single statement, 
    and is generally discouraged in Pandas because it can create a copy of the data that is difficult to track, 
    leading to unexpected behavior when modifying the original data.

    By setting 'mode.chained_assignment' to None, 
    Pandas will no longer issue a warning when this occurs.
    However, it is still generally a good practice to avoid chained indexing and 
    instead use explicit .loc or .iloc accessor methods to modify a DataFrame.

"""


positions = node_features["str_positions"].str.split(",", expand=True) # spliting based on str_position then saving each on its seperate column

# print(positions)

node_features["first_position"] = positions[0]
# print(node_features["first_position"])
# One-hot encoding
node_features = pd.concat([node_features, pd.get_dummies(node_features["first_position"])], axis=1, join='inner') 
# pd.get_dummies() method to perform one-hot encoding on the "first_position" column and concatenate the resulting columns to the original node_features DataFrame using pd.concat() method.
"""
One hot encoding is a process of representing categorical data in a binary format that can be easily processed by machine learning models. It is a method of converting categorical variables into a numerical format.

In one hot encoding, each category is represented by a binary vector with the length equal to the number of categories. In this binary vector, only one bit is set to 1, while all other bits are set to 0. The position of the bit that is set to 1 corresponds to the category of the variable.

For example, if we have a categorical variable "color" with three categories: "red", "green", and "blue", then one hot encoding will represent each category with a binary vector as follows:

"red": [1, 0, 0]
"green": [0, 1, 0]
"blue": [0, 0, 1]
This representation allows machine learning models to easily understand and process categorical data as numeric inputs.

"""

node_features.drop(["str_positions", "first_position"], axis=1, inplace=True)
print( node_features.head() ) # node features after making changes

##################################################################################################################
#       Converting to Numpy
##################################################################################################################

# Convert to numpy as most of lib supports numpy
x = node_features.to_numpy()
print(x.shape) # [num_nodes x num_features] = (5023, 18)

# Sort to define the order of nodes
sorted_df = fifa_df.sort_values(by="int_player_id")
# Select node features
labels = sorted_df[["int_overall"]]
print(labels.head())

# Convert to numpy
y = labels.to_numpy()
print(y.shape) # [num_nodes, 1] --> node regression

#################################################################################################################
#       Extracting the Edges
##################################################################################################################

# Remap player IDs
fifa_df["int_player_id"] = fifa_df.reset_index().index

# just creating permutaion inside the team so that each player is coneected with every other player

# This tells us how many players per team we have to connect
fifa_df["str_team_name"].value_counts()


import itertools
import numpy as np

teams = fifa_df["str_team_name"].unique()   #creating list of teams 
all_edges = np.array([], dtype=np.int32).reshape((0, 2))    # This line initializes an empty numpy array all_edges of size 0x2 with data type int32. It will be used to store the edges of the graph. 
for team in teams:  # for each and every team create permutation to join each and every player
    team_df = fifa_df[fifa_df["str_team_name"] == team]
    players = team_df["int_player_id"].values
    # Build all combinations, as all players are connected
    permutations = list(itertools.combinations(players, 2))
    edges_source = [e[0] for e in permutations]
    edges_target = [e[1] for e in permutations]
    team_edges = np.column_stack([edges_source, edges_target]) #horizontal stack
    all_edges = np.vstack([all_edges, team_edges])  #vertical stack
# Convert to Pytorch Geometric format
edge_index = all_edges.transpose()      # edge index will be created in COO format
print(edge_index) # [2, num_edges]

#################################################################################################################
#       Building The DataSet
##################################################################################################################



from torch_geometric.data import Data
data = Data(x=x, edge_index=edge_index, y=y)

# x = > node feature matrix of football player
# edge_index = indices
# y = label 

print(data)

from torch_geometric.loader import DataLoader
data_list = [Data(...), ..., Data(...)]
print(data_list)
loader = DataLoader(data_list, batch_size=32)

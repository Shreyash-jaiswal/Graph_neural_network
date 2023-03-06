import pandas as pd
# wget -q http://s3.amazonaws.com/tripdata/201306-citibike-tripdata.zip
# unzip -o 201306-citibike-tripdata.zip

trips = pd.read_csv("201306-citibike-tripdata.csv")  
trips.head()

cols_to_drop = ["start station name", "end station name", ]
trips.dropna(inplace=True)
trips.drop(cols_to_drop, axis=1, inplace=True)
trips.head()

# Reassign the location IDs (makes it easier later, because here the IDs didn't start at 0)
locations = trips['start station id'].unique()
new_ids = list(range(len(trips['start station id'].unique())))
mapping = dict(zip(locations, new_ids))

trips['start station id'] = trips['start station id'].map(mapping)
trips['end station id'] = trips['end station id'].map(mapping)
trips.head()

trips = trips.sort_values(by="starttime")
trips.head()

from datetime import datetime, timedelta
import seaborn as sns
sns.set(rc={'figure.figsize':(20,6)})

# Convert columns to datetime
trips["starttime"] = pd.to_datetime(trips["starttime"], format="%Y-%m-%d %H:%M:%S")
trips["stoptime"] = pd.to_datetime(trips["stoptime"], format="%Y-%m-%d %H:%M:%S")

start_date = datetime.strptime("2013-06-01 00:00:01", "%Y-%m-%d %H:%M:%S")
end_date = datetime.strptime("2013-07-01 00:10:34", "%Y-%m-%d %H:%M:%S")
interval = timedelta(minutes=60)
bucket_elements = []
while start_date <= end_date:
    # Check how many trips fall into this interval
    bucket_elements.append(trips[((start_date + interval) >= trips["stoptime"])
                                  & (start_date <= trips["stoptime"])].shape[0])
    # Increment
    start_date += interval

sns.scatterplot(x="index", y="trips_per_hour", data=pd.DataFrame(bucket_elements, columns=["trips_per_hour"]).reset_index())




import numpy as np
# Find out how many outgoing bikers we have
outgoing_trips = trips.groupby("start station id").count()["bikeid"].values
incoming_trips = trips.groupby("end station id").count()["bikeid"].values

# Normalize features between 0 and 1
outgoing_trips = (outgoing_trips - np.min(outgoing_trips)) / (np.max(outgoing_trips) - np.min(outgoing_trips))
incoming_trips = (incoming_trips - np.min(incoming_trips)) / (np.max(incoming_trips) - np.min(incoming_trips))

# Build node features
node_features = np.stack([outgoing_trips, incoming_trips]).transpose()
print("Full shape: ", node_features.shape)
node_features[:10] # [num_nodes x num_features]


from sklearn.utils.extmath import cartesian
from geopy.distance import geodesic

# Get all possible start locations and their geo info
subset = ["start station longitude", "start station latitude", "start station id"]
all_starts = trips.drop_duplicates(subset="start station id", keep="first")[subset]
# Get all possible end locations and their geo info
subset = ["end station longitude", "end station latitude", "end station id"]
all_ends = trips.drop_duplicates(subset="end station id", keep="first")[subset]
# Combine all combinations in one dataframe
distance_matrix = all_ends.merge(all_starts, how="cross")
distance_matrix["distance"] = distance_matrix.apply(lambda x: geodesic((x["start station latitude"], x["start station longitude"]), 
                                                          (x["end station latitude"], x["end station longitude"])).meters, axis=1)
distance_matrix.head()



distance_matrix["edge"] = distance_matrix["distance"] < 500
distance_matrix.head()


# Use mask to extract static edges
edge_index = distance_matrix[distance_matrix["edge"] == True][["start station id", "end station id"]].values
edge_index = edge_index.transpose()
edge_index # [2 x num_edges]

# Add edge features to indicate edge type
distance_feature = distance_matrix[distance_matrix["edge"] == True]["distance"].values
edge_type_feature = np.zeros_like(distance_feature) # 0 = static edge
trip_duration_feature = np.zeros_like(distance_feature) # 0 = no information
static_edge_features = np.stack([distance_feature, edge_type_feature, trip_duration_feature]).transpose()
static_edge_features # [num_edges x num_features]

def extract_dynamic_edges(s):
    # Extract dynamic edges and their features
    trip_indices = s[["start station id", "end station id"]].values
    trip_durations = s["tripduration"]

    # Build edge features
    distance_feature  = pd.DataFrame(trip_indices, 
                                    columns=["start station id", "end station id"]).merge(
                                        distance_matrix, on=["start station id", "end station id"], 
                                        how="left")["distance"].values
    edge_type_feature = np.ones_like(distance_feature) # 1 = dynamic
    trip_duration_feature = trip_durations
    edge_features = np.stack([distance_feature, edge_type_feature, trip_duration_feature]).transpose()
    return edge_features, trip_indices.transpose()



start_date = datetime.strptime("2013-06-01 00:00:01", "%Y-%m-%d %H:%M:%S")
end_date = datetime.strptime("2013-07-01 00:10:34", "%Y-%m-%d %H:%M:%S")
interval = timedelta(minutes=60)

xs = []
edge_indices = []
ys = []
y_indices = []
edge_features = []


while start_date <= end_date:
    # 0 - 60 min 
    current_snapshot = trips[((start_date + interval) >= trips["stoptime"])
                                  & (start_date <= trips["stoptime"])]
    # 60 - 120 min
    subsequent_snapshot = trips[((start_date + 2*interval) >= trips["stoptime"])
                                  & (start_date + interval <= trips["stoptime"])]
    # Average duplicate trips
    current_snapshot = current_snapshot.groupby(["start station id", "end station id"]).mean().reset_index()
    subsequent_snapshot = subsequent_snapshot.groupby(["start station id", "end station id"]).mean().reset_index()

    # Extract dynamic trip edges
    edge_feats, additional_edge_index = extract_dynamic_edges(current_snapshot)
    exteneded_edge_index = np.concatenate([edge_index, additional_edge_index], axis=1)
    extended_edge_feats = np.concatenate([edge_feats, static_edge_features], axis=0)

    # Labels
    y = subsequent_snapshot["tripduration"].values
    y_index = subsequent_snapshot[["start station id", "end station id"]].values

    # Append everything
    xs.append(node_features) # static
    edge_indices.append(exteneded_edge_index) # static + dynamic
    edge_features.append(extended_edge_feats) # static + dynamic
    ys.append(y) # dynamic
    y_indices.append(y_index.transpose()) # dynamic

    # Increment
    start_date += interval


i = 2
print(f"""Example of graph snapshot {i}: \n
      Node feature shape: {xs[i].shape} \n
      Edge index shape: {edge_indices[i].shape} \n
      Edge feature shape: {edge_features[i].shape} \n 
      Labels shape: {ys[i].shape} \n
      Labels mask shape: {y_indices[i].shape}
      """)


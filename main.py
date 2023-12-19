import os
import json
import base64
import pandas as pd
import requests
from math import radians, cos, sin, asin, sqrt
from google.cloud import pubsub_v1
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from geopy.distance import geodesic

pubsub = pubsub_v1.PublisherClient()

def publish(topicName, data):
  dataStr = json.dumps(data)
  dataBuffer = dataStr.encode("utf-8")
  topic_path = pubsub.topic_path(os.getenv('GCP_PROJECT'), topicName)
  future = pubsub.publish(topic_path, dataBuffer)
  print(future.result())
  print(f"Published messages to {topic_path}.")

def pushFcm(deviceToken, tenNearestPlace):
    api_url = "https://fcm.googleapis.com/fcm/send"
    pushData = {
        "data": {
             "tenNearestPlace": tenNearestPlace, 
              "notification": {
                "title": "FCM MESSAGE TEN NEAREST PLACE",
                "body": "Berhasil mendapatkan 10 rekomendasi tempat terdekat",
                "icon": "",
              }
        },
        "to": deviceToken
    }
    headers =  {
        "Authorization": "key="+os.getenv('FCM_KEY'),
        "Content-Type": "application/json"
    }
    response = requests.post(api_url, data=json.dumps(pushData), headers=headers)
    print(response.status_code)

def search(event, context):
    print("search request ")
    dataStr = base64.b64decode(event["data"]).decode("utf-8")
    data = json.loads(dataStr)

    print("request "+data)

    dataset_new = pd.read_csv("toor.csv")
    location_recommender_tf = SearchModel(dataset_new)

    place_name = data['placename']

    recommendations_by_place = location_recommender_tf.recommend_by_place_name(place_name)
    if recommendations_by_place is not None:
        recommendations_by_place

    json_records = recommendations_by_place.to_json(orient ='records')

    messageData =  { "data" : json_records }
    print(messageData)

    pushFcm(data["devicetoken"], json_records)

def nearbyTreasure(event, context):
    print("nearby treasure")
    dataStr = base64.b64decode(event["data"]).decode("utf-8")
    data = json.loads(dataStr)

    print("request "+data)

    dataset_new = pd.read_csv("toor.csv")
    location_recommender_tf = NearbyTreasureModel(dataset_new)

    user_coordinates = (data['lat'], data['lon'])
    recommendations_user = location_recommender_tf.recommend_nearby_location(user_coordinates)
    
    json_records = recommendations_user.to_json(orient ='records')

    messageData =  { "data" : json_records }
    print(messageData)

    pushFcm(data["devicetoken"], json_records)

def topDestination(event, context):
    print("top destination")
    dataStr = base64.b64decode(event["data"]).decode("utf-8")
    data = json.loads(dataStr)

    print("request "+data)

    dataset_new = pd.read_csv("toor.csv")
    location_recommender_tf = TopDestinationModel(dataset_new)

    user_coordinates = (data['lat'], data['lon'])
    recommendations_user = location_recommender_tf.recommend_nearby_location(user_coordinates)
    
    json_records = recommendations_user.to_json(orient ='records')

    messageData =  { "data" : json_records }
    print(messageData)

    pushFcm(data["devicetoken"], json_records)
    
def searchByCategory(event, context):
    print("top destination")
    dataStr = base64.b64decode(event["data"]).decode("utf-8")
    data = json.loads(dataStr)

    print("request "+data)

    dataset_new = pd.read_csv("toor.csv")
    location_recommender_tf = SearchByCategoryModel(dataset_new)

    user_coordinates = (data['lat'], data['lon'])
    user_category = data['usercategory']

    recommendations_user = location_recommender_tf.recommend_nearby_location(user_coordinates, user_category, max_distance=100)
  
 
    json_records = recommendations_user.to_json(orient ='records')

    messageData =  { "data" : json_records }
    print(messageData)

    pushFcm(data["devicetoken"], json_records)

class SearchModel:
    def __init__(self, dataset):
        self.locations = dataset[['Place_Name', 'Description', 'Rating', 'Category', 'Photos', 'Lat', 'Long']]
        self.recommendations_user = None
        self.train_model()

    def train_model(self):
        location_vectors = self.locations[['Lat', 'Long']].values
        merged_vectors = np.concatenate([location_vectors], axis=1)

        input_layer = Input(shape=(merged_vectors.shape[1],))
        encoded = Dense(128, activation='relu')(input_layer)
        decoded = Dense(merged_vectors.shape[1], activation='linear')(encoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        autoencoder.fit(merged_vectors, merged_vectors, epochs=50, batch_size=32, verbose=0)

        self.autoencoder = autoencoder

    def recommend_by_place_name(self, place_name):
        # Cari koordinat dari nama tempat yang dicari
        place_coords = self.locations[self.locations['Place_Name'] == place_name][['Lat', 'Long']].values
        if len(place_coords) == 0:
            print("Tempat wisata tidak ditemukan.")
            return None

        # Hitung jarak antara tempat wisata yang dicari dan lokasi lainnya dalam dataset
        distances = []
        for idx, row in self.locations.iterrows():
            destination_coords = (row['Lat'], row['Long'])
            distance = geodesic(place_coords[0], destination_coords).kilometers
            distances.append((idx, distance))

        # Urutkan berdasarkan jarak dan ambil top 10 lokasi terdekat
        sorted_distances = sorted(distances, key=lambda x: x[1])[:10]

        recommendations = []
        for idx, _ in sorted_distances:
            recommendations.append(self.locations.iloc[idx])

        self.recommendations_user = pd.DataFrame(recommendations)

        return self.recommendations_user

    def save_model(self, filename):
        if self.autoencoder is None:
            print("No model to save.")
            return

        self.autoencoder.save(filename)
        print(f"Model saved successfully as {filename}")

    def load_model(self, filename):
        loaded_model = load_model(filename)
        self.autoencoder = loaded_model
        print(f"Model loaded successfully from {filename}")

class NearbyTreasureModel:
    def __init__(self, dataset):
        self.locations = dataset[['Place_Name', 'Description', 'Rating', 'Category', 'Photos', 'Lat', 'Long']]
        self.recommendations_user = None
        self.train_model()

    def train_model(self):
        location_vectors = self.locations[['Lat', 'Long']].values
        merged_vectors = np.concatenate([location_vectors], axis=1)

        input_layer = Input(shape=(merged_vectors.shape[1],))
        encoded = Dense(128, activation='relu')(input_layer)
        decoded = Dense(merged_vectors.shape[1], activation='linear')(encoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        autoencoder.fit(merged_vectors, merged_vectors, epochs=50, batch_size=32, verbose=0)

        self.autoencoder = autoencoder

    def recommend_nearby_location(self, user_coords, k=10):
        distances = []
        for idx, row in self.locations.iterrows():
            destination_coords = (row['Lat'], row['Long'])
            distance = geodesic(user_coords, destination_coords).kilometers
            distances.append((idx, distance))

        distances = sorted(distances, key=lambda x: x[1])[:k]

        recommendations = []
        for idx, dist in distances:
            recommendations.append(self.locations.iloc[idx])

        self.recommendations_user = pd.DataFrame(recommendations)

        return self.recommendations_user

    def save_model_with_recommendations(self, filename):
        if self.recommendations_user is None:
            print("No recommendations to save.")
            return

        self.autoencoder.save(filename)
        self.recommendations_user.to_csv(f"{filename}_recommendations.csv", index=False)  # Save recommendations to a CSV file
        print(f"Model saved successfully as {filename} with recommendations")

    def load_model_with_recommendations(self, filename):
        loaded_model = load_model(filename)
        self.autoencoder = loaded_model
        recommendations_file = f"{filename}_recommendations.csv"
        self.recommendations_user = pd.read_csv(recommendations_file) if os.path.exists(recommendations_file) else None
        print(f"Model loaded successfully from {filename} with recommendations")

class TopDestinationModel:
    def __init__(self, dataset):
        self.locations = dataset[['Place_Name', 'Description', 'Rating', 'Category', 'Photos', 'Lat', 'Long']]
        self.recommendations_user = None
        self.train_model()

    def train_model(self):
        location_vectors = self.locations[['Lat', 'Long']].values
        merged_vectors = np.concatenate([location_vectors], axis=1)

        input_layer = Input(shape=(merged_vectors.shape[1],))
        encoded = Dense(128, activation='relu')(input_layer)
        decoded = Dense(merged_vectors.shape[1], activation='linear')(encoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        autoencoder.fit(merged_vectors, merged_vectors, epochs=50, batch_size=32, verbose=0)

        self.autoencoder = autoencoder

    def recommend_nearby_location(self, user_coords, k=10):
        distances = []
        for idx, row in self.locations.iterrows():
            destination_coords = (row['Lat'], row['Long'])
            distance = geodesic(user_coords, destination_coords).kilometers
            distances.append((idx, distance))

        distances = sorted(distances, key=lambda x: x[1])[:k]

        recommendations = []
        for idx, dist in distances:
            recommendations.append(self.locations.iloc[idx])

        recommendations_df = pd.DataFrame(recommendations)
        sorted_recommendations = recommendations_df.sort_values(by='Rating', ascending=False)  # Urutkan berdasarkan rating tertinggi
        self.recommendations_user = sorted_recommendations

        return self.recommendations_user

    def save_model_with_recommendations(self, filename):
        if self.recommendations_user is None:
            print("No recommendations to save.")
            return

        self.autoencoder.save(filename)
        self.recommendations_user.to_csv(f"{filename}_recommendations.csv", index=False)  # Save recommendations to a CSV file
        print(f"Model saved successfully as {filename} with recommendations")

    def load_model_with_recommendations(self, filename, user_coordinates):
        loaded_model = load_model(filename)
        self.autoencoder = loaded_model
        recommendations_file = f"{filename}_recommendations.csv"
        self.recommendations_user = pd.read_csv(recommendations_file) if os.path.exists(recommendations_file) else None
        print(f"Model loaded successfully from {filename} with recommendations")

        if self.recommendations_user is not None:
            # Jika model dimuat dengan rekomendasi sebelumnya, lakukan pengurutan kembali berdasarkan rating tertinggi
            self.recommendations_user = self.recommend_nearby_location(user_coordinates)
            print("Recommendations sorted by rating after loading the model.")
            return self.recommendations_user
        
class SearchByCategoryModel:
    def __init__(self, dataset):
        self.locations = dataset[['Place_Name', 'Description', 'Rating', 'Category', 'Photos', 'Lat', 'Long']]
        self.recommendations_user = None
        self.train_model()

    def train_model(self):
        location_vectors = self.locations[['Lat', 'Long']].values
        merged_vectors = np.concatenate([location_vectors], axis=1)

        input_layer = Input(shape=(merged_vectors.shape[1],))
        encoded = Dense(128, activation='relu')(input_layer)
        decoded = Dense(merged_vectors.shape[1], activation='linear')(encoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        autoencoder.fit(merged_vectors, merged_vectors, epochs=50, batch_size=32, verbose=0)

        self.autoencoder = autoencoder

    def recommend_nearby_location(self, user_coords, user_category=None, k=10, max_distance=100):
        distances = []
        for idx, row in self.locations.iterrows():
            destination_coords = (row['Lat'], row['Long'])
            distance = geodesic(user_coords, destination_coords).kilometers

            if user_category and distance <= max_distance:
                if row['Category'] == user_category:
                    distances.append((idx, distance))
            elif not user_category and distance <= max_distance:
                distances.append((idx, distance))

        distances = sorted(distances, key=lambda x: x[1])[:k]

        recommendations = []
        for idx, dist in distances:
            recommendations.append(self.locations.iloc[idx])

        recommendations_df = pd.DataFrame(recommendations)
        sorted_recommendations = recommendations_df.sort_values(by='Rating', ascending=False)
        self.recommendations_user = sorted_recommendations

        return self.recommendations_user

    def save_model_with_recommendations(self, filename):
        if self.recommendations_user is None:
            print("No recommendations to save.")
            return

        self.autoencoder.save(filename)
        self.recommendations_user.to_csv(f"{filename}_recommendations.csv", index=False)
        print(f"Model saved successfully as {filename} with recommendations")

    def load_model_with_recommendations(self, filename, user_coordinates, user_category=None):
        loaded_model = load_model(filename)
        self.autoencoder = loaded_model
        recommendations_file = f"{filename}_recommendations.csv"
        self.recommendations_user = pd.read_csv(recommendations_file) if os.path.exists(recommendations_file) else None
        print(f"Model loaded successfully from {filename} with recommendations")

        if self.recommendations_user is not None:
            self.recommendations_user = self.recommend_nearby_location(user_coordinates, user_category)
            print("Recommendations sorted by rating after loading the model.")
            return self.recommendations_user
        


# Function Not USED  
places = pd.read_csv('datasets/tourism_with_id.csv')
places = places.drop(['Description', 'Time_Minutes', 'Coordinate', 'Unnamed: 11', 'Unnamed: 12'], axis=1)

def dist(lat1, long1, lat2, long2):
    """ Replicating the same formula as mentioned in Wiki """
    # convert decimal degrees to radians
    lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])
    # haversine formula
    dlon = long2 - long1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

def find_nearest(lat, long):
    distances = places.apply(
        lambda row: dist(lat, long, row['Lat'], row['Long']),
        axis=1)
    return distances.sort_values(ascending = True).head(10)

def get_10_nearest_place(event, context):
    print(event['data'])
    dataStr = base64.b64decode(event["data"]).decode("utf-8")
    print(dataStr)
    data = json.loads(dataStr)
    print(data)

    tenNearestPlace = find_nearest(float(data['lat']), float(data['lon']))
    df2= places.filter(items = tenNearestPlace.index, axis=0)
    json_records = df2.to_json(orient ='records')

    # topicName = os.getenv('RESULT_TOPIC')
    messageData =  { "data" : json_records }
    print(messageData)

    #publish(topicName, messageData)
    pushFcm(data["devicetoken"], json_records)
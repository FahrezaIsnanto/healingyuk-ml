import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from geopy.distance import geodesic

class LocationRecommenderTensorFlow:
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

# Contoh penggunaan
dataset_new = pd.read_csv("toor.csv")
location_recommender_tf = LocationRecommenderTensorFlow(dataset_new)

# Penggunaan untuk mencari rekomendasi tempat wisata berdasarkan nama tempat
place_name = "Dunia Fantasi"

recommendations_by_place = location_recommender_tf.recommend_by_place_name(place_name)
if recommendations_by_place is not None:
    recommendations_by_place

# Simpan model
# location_recommender_tf.save_model("Search_location.h5")
json_records = recommendations_by_place.to_json(orient ='records')
print(json_records)

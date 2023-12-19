import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from geopy.distance import geodesic
import os

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

# Example usage
dataset_new = pd.read_csv("toor.csv")
location_recommender_tf = LocationRecommenderTensorFlow(dataset_new)

user_coordinates = (-0.307619325290053, 100.36425844607963)
recommendations_user = location_recommender_tf.recommend_nearby_location(user_coordinates)
#location_recommender_tf.save_model_with_recommendations("Top_destination.h5")
print(recommendations_user)

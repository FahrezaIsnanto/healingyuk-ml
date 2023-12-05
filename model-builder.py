import pandas as pd
import joblib
from math import radians, cos, sin, asin, sqrt
import dill as pickle

# Import dataset
places = pd.read_csv('datasets/tourism_with_id.csv')

# Prepare dataset
places = places.drop(['Description', 'Time_Minutes', 'Coordinate', 'Unnamed: 11', 'Unnamed: 12'], axis=1)

# Creating Model

# 1. Creating Haversine Formula for counting distance between 2 latitude longitude
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

# 2. Creating function to Find 10 nearest place 
def get10NearestPlace(lat, long):
    distances = places.apply(
        lambda row: dist(lat, long, row['Lat'], row['Long']),
        axis=1)
    return distances.sort_values(ascending = True).head(10)

# 3. Creating final model
def model(lat, long):
    tenNearestPlace = get10NearestPlace(lat, long)
    return places.filter(items = tenNearestPlace.index, axis=0)

print(model(-7.016901205125758, 110.46777246024239))

# 4. Deploy model
pickle.settings['recurse'] = True
with open('models/healingyuk.pkl', 'wb') as file:
    pickle.dump(model,file)
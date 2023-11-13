import os
import json
import base64
import pandas as pd
import requests
from math import radians, cos, sin, asin, sqrt
from google.cloud import pubsub_v1

pubsub = pubsub_v1.PublisherClient()

places = pd.read_csv('datasets/tourism_with_id.csv')
places = places.drop(['Description', 'Time_Minutes', 'Coordinate', 'Unnamed: 11', 'Unnamed: 12'], axis=1)

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

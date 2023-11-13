import json
from google.cloud import pubsub_v1

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path("fahrezaisnantodev", "process_topic")

def publish():
  dataStr = json.dumps({"lat":-7.016901205125758,"lon":110.46777246024239})
  dataBuffer = dataStr.encode("utf-8")
  future = publisher.publish(topic_path, dataBuffer)
  print(future.result())

publish()
print(f"Published messages to {topic_path}.")


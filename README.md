gcloud functions deploy machine-learning \  
--runtime python311 \
--trigger-topic process_topic \
--entry-point get_10_nearest_place \
--set-env-vars "GCP_PROJECT=fahrezaisnantodev,RESULT_TOPIC=result_topic"

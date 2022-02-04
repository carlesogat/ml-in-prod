#!/bin/bash

# exit when any command fails
set -e

INPUT_DATA=gs://class-ml-prod-2022/data/raw/aclImdb
OUTPUT_DATA=gs://ferrous-linker-339609/output/

NUM_WORKERS=5
WORKER_TYPE=n2-standard-8

REGION=europe-west4
PROJECT=ferrous-linker-339609
TEMP_LOCATION=gs://ferrous-linker-339609/tmp

CONTAINER=$REGION-docker.pkg.dev/$PROJECT/dataflow-containers/ml-in-prod-container

python pipeline/preprocess_pipeline.py \
  --runner=DataflowRunner \
  --region=$REGION \
  --project=$PROJECT \
  --temp_location=$TEMP_LOCATION \
  --experiments=use_runner_v2 \
  --sdk_container_image=$CONTAINER \
  --worker_machine_type=$WORKER_TYPE \
  --autoscaling_algorithm=NONE \
  --num_workers=$NUM_WORKERS \
  --data-location=$INPUT_DATA \
  --output-location=$OUTPUT_DATA
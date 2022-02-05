export AIP_MODEL_DIR=training-output/

mkdir -p $AIP_MODEL_DIR

python trainer/task.py \
    --data-location=gs://bucket_olg_01/output/ \
    --tft-location=gs://bucket_olg_01/output/transform_fn/ \
    --batch-size=1024 \
    --epochs=10
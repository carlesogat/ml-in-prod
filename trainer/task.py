"""A simple main file to showcase the template."""

import logging
import os
import sys
from typing import List, Dict, Union
from urllib.parse import urlparse

import tensorflow as tf
import tensorflow_transform as tft
from google.cloud import storage
from google.cloud.storage import Blob
from hypertune import hypertune
from keras.layers import TextVectorization
from keras import layers
from keras import activations
from keras import models
from keras import losses
from keras import metrics
from keras.optimizer_v2.rmsprop import RMSProp


def read_dataset(filenames: List[str], feature_spec) -> tf.data.TFRecordDataset:
    dataset: tf.data.TFRecordDataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda r: tf.io.parse_single_example(r, feature_spec))
    dataset = dataset.map(lambda d: (d['text'], d['target']))
    return dataset


def get_filename_list(file_pattern: str) -> List[str]:
    sc = storage.Client()
    url_parts = urlparse(file_pattern)
    bucket_name = url_parts.hostname
    location = url_parts.path[1:]
    output: List[Blob] = sc.list_blobs(bucket_name, prefix=location)
    paths: List[str] = [f"gs://{b.bucket.name}/{b.name}" for b in output]
    return paths


def build_model():
    inputs = layers.Input(shape=[20000])
    x = layers.Dense(256, activation=activations.relu)(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation=activations.sigmoid)(x)
    model = models.Model(inputs, outputs, name="my-kschool-model")
    model.compile(optimizer=RMSProp(), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
    return model


def train_and_evaluate(data_location: str,
                       tft_location: str,
                       batch_size: int,
                       epochs: int):
    train_location = os.path.join(data_location, "train/")
    test_location = os.path.join(data_location, "test/")

    tft_output = tft.TFTransformOutput(tft_location)
    feature_spec = tft_output.transformed_feature_spec()

    filenames_train = get_filename_list(train_location)
    filenames_test = get_filename_list(test_location)

    train_ds: tf.data.TFRecordDataset = read_dataset(filenames_train, feature_spec)
    test_ds: tf.data.TFRecordDataset = read_dataset(filenames_test, feature_spec)

    x_train_text = train_ds.map(lambda text, target: text)

    vectorizer = TextVectorization(ngrams=2, max_tokens=20000)
    vectorizer.adapt(x_train_text)

    train_ds = train_ds.map(lambda text, target: (vectorizer(text), target))
    test_ds = test_ds.map(lambda text, target: (vectorizer(text), target))

    train_ds = train_ds.batch(batch_size=batch_size)
    test_ds = test_ds.batch(batch_size=batch_size)

    model = build_model()
    model.summary(print_fn=logging.info)

    model.fit(train_ds, epochs=epochs, validation_data=test_ds)
    loss, acc = model.evaluate(test_ds)
    logging.info(f"LOSS: {loss:.4f}")
    logging.info(f"ACC: {acc:.4f}")

    metric_tag = "kschool_accuracy"
    ht = hypertune.HyperTune()
    ht.report_hyperparameter_tuning_metric(hyperparameter_metric_tag=metric_tag,
                                           metric_value=acc,
                                           global_step=epochs)

    model_dir = os.environ.get('AIP_MODEL_DIR')
    model.save(model_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--data-location', default=None, required=True)
    parser.add_argument('--tft-location', required=True)
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--epochs', required=True, type=int)

    args = parser.parse_args()

    loglevel = 'INFO'
    logging.basicConfig(stream=sys.stdout, level=loglevel)

    train_and_evaluate(args.data_location,
                       args.tft_location,
                       args.batch_size,
                       args.epochs)
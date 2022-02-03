import os

from typing import List, Dict

import apache_beam as beam
import tensorflow_transform.beam as tft_beam
import tensorflow_transform as tft
import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata, schema_utils
from apache_beam import PCollection, Pipeline
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions
from tfx_bsl.cc.tfx_bsl_extension.coders import RecordBatchToExamples


def get_train_and_test(p: Pipeline, data_location: str) -> (PCollection[Dict], PCollection[Dict]):
    train_pos_location = os.path.join(data_location, "train/pos/")
    train_neg_location = os.path.join(data_location, "train/neg")
    test_pos_location = os.path.join(data_location, "test/pos/")
    test_neg_location = os.path.join(data_location, "test/neg/")

    train_pos: PCollection[str] = p | "Train pos" >> beam.io.ReadFromText(train_pos_location)
    train_neg: PCollection[str] = p | "Train neg" >> beam.io.ReadFromText(train_neg_location)

    test_pos: PCollection[str] = p | "Test pos" >> beam.io.ReadFromText(test_pos_location)
    test_neg: PCollection[str] = p | "Test neg" >> beam.io.ReadFromText(test_neg_location)

    train_pos_dicts: PCollection[Dict] = train_pos | "Add label train pos" >> beam.Map(
        lambda t: {'text': t, 'target': 1})
    train_neg_dicts: PCollection[Dict] = train_neg | "Add label train neg" >> beam.Map(
        lambda t: {'text': t, 'target': 0})

    train_dicts: PCollection[Dict] = (train_pos_dicts, train_neg_dicts) | "Train set" >> beam.Flatten()

    test_pos_dicts = test_pos | "Add label test pos" >> beam.Map(
        lambda t: {'text': t, 'target': 1})
    test_neg_dicts = test_neg | "Add label test neg" >> beam.Map(
        lambda t: {'text': t, 'target': 0})

    test_dicts = (test_pos_dicts, test_neg_dicts) | "Test set" >> beam.Flatten()

    return train_dicts, test_dicts


def preprocessing_fn_tfidf(inputs):
    # Just an example, not used here

    texts = inputs['text']
    targets = inputs['target']

    # "Hola que peli mas molona"
    words = tf.strings.split(texts, sep=" ").to_sparse()
    # ["Hola", "que", ....]
    ngrams = tft.ngrams(words, ngram_range=(1, 2), separator=" ")
    vocabulary = tft.compute_and_apply_vocabulary(ngrams, top_k=20000)
    indices, weights = tft.tfidf(vocabulary, 20000)

    return {'indices': indices, 'weights': weights, 'targets': targets}


def preprocessing_fn_1_hot(inputs):
    # Just an example, not used here

    texts = inputs['text']
    targets = inputs['target']

    words = tf.strings.split(texts, sep=" ").to_sparse()
    vocabulary = tft.compute_and_apply_vocabulary(words, top_k=20000)
    hot_encoded_vector = tf.one_hot(vocabulary, depth=20000, on_value=1, off_value=0, dtype=tf.int8)

    ### How to read later if you import these TFRecords with Keras for training
    # tf.sparse.to_dense(hot_encoded_vector, default_value=0)

    return {'hot_encoded': hot_encoded_vector, 'targets': targets}


def preprocessing_fn(inputs):
    # texts = inputs['text']
    # targets = inputs['target']

    outputs = inputs.copy()

    return outputs


def apply_tensorflow_transform(train_set: PCollection[Dict], test_set: PCollection[dict], metadata):
    transf_train_ds, transform_fn = (train_set, metadata) | "TFT train" >> \
                                    tft_beam.AnalyzeAndTransformDataset(
                                        preprocessing_fn=preprocessing_fn,
                                        output_record_batches=True)

    transf_train_pcoll, _ = transf_train_ds

    test_set_ds = (test_set, metadata)

    transf_test_ds = \
        (test_set_ds, transform_fn) | "TFT test" >> tft_beam.TransformDataset(output_record_batches=True)

    transf_test_pcoll, _ = transf_test_ds

    return transf_train_pcoll, transf_test_pcoll, transform_fn


def run_pipeline(argv: List[str], data_location: str, output_location: str):
    feature_spec = {
        'text': tf.io.FixedLenFeature([], tf.strings),
        'target': tf.io.FixedLenFeature([], tf.int64)
    }

    metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_as_feature_spec(feature_spec)
    )

    options = PipelineOptions(argv)
    gcp_options = options.view_as(GoogleCloudOptions)
    temp_dir = gcp_options.temp_location

    train_output_location = os.path.join(output_location, "train/")
    test_output_location = os.path.join(output_location, "test/")

    with beam.Pipeline(options=options) as p, tft_beam.Context(temp_dir=temp_dir):
        train_set, test_set = get_train_and_test(p, data_location)
        train_set_transf, test_set_transf, transform_fn = apply_tensorflow_transform(train_set, test_set, metadata)

        # PCollection[Listas[elementos]] --> PCollection[elementos]
        #    3 listas de 15 elems                  45 elems

        train_set_tf_example: PCollection[tf.train.Example] = train_set_transf | "Train to example" >> beam.FlatMap(
            lambda r, _: RecordBatchToExamples(r))

        train_set_tf_example | "Write train" >> beam.io.WriteToTFRecord(file_path_prefix=train_output_location,
                                                                        file_name_suffix=".tfrecord")

        test_set_tf_example = test_set_transf | "Text to example" >> beam.FlatMap(
            lambda r, _: RecordBatchToExamples(r))

        test_set_tf_example | "Write test" >> beam.io.WriteToTFRecord(file_path_prefix=test_output_location,
                                                                      file_name_suffix=".tfrecord")

        transform_fn_location = os.path.join(output_location, "transform_fn/")
        transform_fn | "Write transform fn" >> tft_beam.WriteTransformFn(transform_fn_location)

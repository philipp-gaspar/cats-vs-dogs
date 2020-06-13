"""Example implementation of code to run on the Cloud ML service.
"""

import argparse
import logging
import json
import os

import tensorflow as tf
import trainer.deep_models as deep_models

if __name__ == '__main__':
    print('Using Tensorflow version: %s' % str(tf.version.VERSION))
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--bucket',
        action='store',
        required=True,
        help='Training data will be in gs://BUCKET/cats_vs_dogs/data/train/')

    parser.add_argument(
        '--job-dir',
        action='store',
        default='./junk',
        help='Requiered by gcloud.')

    parser.add_argument(
        '--epochs',
        aciton='store',
        type=int,
        default=30,
        help='Number of training epochs.')

    parser.add_argument(
        '--batch_size',
        action='store',
        type=int,
        default=64,
        help='Number of examples to compute gradient on')

    # parse args
    args = parser.parse_args()
    args.output_dir = 'gs://%s/cats_vs_dogs/trained_models' % (args.bucket)

    # run
    logging.basicConfig(level=logging.INFO)
    deep_classifier = deep_models.DeepClassifier(args)

    deep_classifier.run_on_cloud()

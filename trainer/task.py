"""Example implementation of code to run on the Cloud ML service.
"""

import argparse
import logging
import json
import os

import tensorflow as tf
import trainer.model as model

if __name__ == '__main__':
    print('Using Tensorflow version: %s' % str(tf.version.VERSION))
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--bucket',
        help='Training data will be in gs://BUCKET/cats_vs_dogs/data/train/',
        required=True)
    parser.add_argument(
        '--job-dir',
        help='Requiered by gcloud.',
        default='./junk')
    parser.add_argument(
        '--train_batch_size',
        help='Number of examples to compute gradient on',
        type=int,
        default=64)

    # parse args
    args = parser.parse_args()
    arguments = args.__dict__

    # set appropriate output directory
    BUCKET = arguments['bucket']
    output_dir = 'gs://%s/cats_vs_dogs/trained_models' % (BUCKET)
    print('Writing trained model to: %s\n' % output_dir)
    arguments['output_dir'] = output_dir

    # run
    logging.basicConfig(level=logging.INFO)
    model.setup(arguments)
    model.train_and_evaluate()

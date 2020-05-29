import logging
import os
import time
import tensorflow as tf
import numpy as np

from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DeepClassifier(object):
    def __init__(self, args):
        self.bucket = args.bucket
        self.job_dir = args.job_dir
        self.batch_size = args.batch_size
        self.output_dir = args.output_dir

        self._setup_cloud_bucket()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
            input_shape=(150, 150, 3)))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

        return model

    def _setup_local_folders(self):
        self.train_dir = os.path.join(os.getcwd(), 'data', 'train')
        self.valid_dir = os.path.join(os.getcwd(), 'data', 'validation')

    def _setup_cloud_bucket(self):
        data_bucket = 'gs://%s/cats_vs_dogs/' % self.bucket
        self.train_dir = os.path.join(data_bucket, 'data', 'train')
        self.valid_dir = os.path.join(data_bucket, 'data', 'validation')

    def _setup_dataset(self):
        train_path = os.path.join(self.train_dir, '*', '*')
        valid_path = os.path.join(self.valid_dir, '*', '*')

        dataset = dict()
        dataset['train'] = tf.data.Dataset.list_files(str(train_path))
        dataset['valid'] = tf.data.Dataset.list_files(str(valid_path))

        return dataset

    def _get_label(self, file_path):
        class_names = 'cats'
        label = tf.strings.split(file_path, os.path.sep)[-2]
        return label == class_names

    def _decode_images(self, img, width=150, height=150):
        tensor = tf.image.decode_jpeg(img, channels=3)
        tensor = tf.image.convert_image_dtype(tensor, tf.float32)
        tensor = tf.image.resize(tensor, [width, height])

        return tensor

    def _process_path(self, file_path):
        label = self._get_label(file_path)
        image_raw = tf.io.read_file(file_path)
        tensor = self._decode_images(image_raw)

        return tensor, label

    def _augment_data(self, image, label):
        image = tf.image.resize_with_crop_or_pad(image, 156, 156)
        image = tf.image.random_crop(image, size=[150, 150, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.5)

        return image, label

    def run_local(self):
        dataset = self._setup_dataset()

        #   Set 'num_parallel_calls' so multiple images are loaded/processed
        # in parallel.
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train_ds = dataset['train'].map(
            self._process_path,
            num_parallel_calls=AUTOTUNE)

        valid_ds = dataset['valid'].map(
            self._process_path,
            num_parallel_calls=AUTOTUNE)

        #   Separate dataset into batches.
        num_examples = tf.data.experimental.cardinality(train_ds).numpy()
        train_batches = train_ds.take(num_examples).cache().shuffle(
            num_examples).map(self._augment_data,
            num_parallel_calls=AUTOTUNE).batch(self.batch_size).prefetch(
            AUTOTUNE)

        valid_batches = valid_ds.batch(2*self.batch_size)

        #   Train the model.
        model = self._build_model()
        history = model.fit(
            train_batches,
            epochs=5,
            validation_data=valid_batches)

        #   Export trained model.
        export_dir = os.path.join(
            self.output_dir,
            'deep_classifier_{}'.format(time.strftime('%Y%m%d-%H%M%S')))
        tf.saved_model.save(model, export_dir)

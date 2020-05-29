import logging
import os
import time
import tensorflow as tf
import numpy as np

from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BUCKET = None
OUTPUT_DIR = None
TRAIN_BATCH_SIZE = None
TRAIN_DATA_DIR = None
VAL_DATA_DIR = None

def setup(args):
    global BUCKET, OUTPUT_DIR
    global TRAIN_BATCH_SIZE, TRAIN_DATA_DIR, VAL_DATA_DIR

    BUCKET = args['bucket']
    OUTPUT_DIR = args['output_dir']
    TRAIN_BATCH_SIZE = int(args['train_batch_size'])

    # set up training and evaluation data patterns
    DATA_BUCKET = 'gs://%s/cats_vs_dogs/' % BUCKET
    # TRAIN_DATA_DIR = os.path.join(DATA_BUCKET, 'data', 'train')     # <=====
    # VAL_DATA_DIR = os.path.join(DATA_BUCKET, 'data', 'validation')  # <=====
    TRAIN_DATA_DIR = os.path.join(os.getcwd(), 'data', 'train')
    VAL_DATA_DIR = os.path.join(os.getcwd(), 'data', 'validation')
    logging.info('Training based on data in: %s' % TRAIN_DATA_DIR)

def deep_classifier():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
        input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(lr=1e-4),
            metrics=['acc'])

    return model

def train_and_evaluate():
    # data augmentation generator
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # the validation data should't be augmented
    val_datagen = ImageDataGenerator(rescale=1./255)

    # generate flow for training and validation data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(150, 150),
        batch_size=TRAIN_BATCH_SIZE,
        class_mode='binary')

    validation_generator = val_datagen.flow_from_directory(
        VAL_DATA_DIR,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    # create the model
    model = deep_classifier()

    # train and evaluate
    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        epochs=5,
                        steps_per_epoch=100,
                        validation_steps=50)

    # export model
    export_dir = os.path.join(OUTPUT_DIR,
        'deep_classifier_{}'.format(time.strftime('%Y%m%d-%H%M%S')))
    print('Exporting to {}'.format(export_dir))
    tf.saved_model.save(model, export_dir)

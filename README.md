# Cats vs Dogs

<p align="center">
<img src="catdog.gif" width="400" ></a>
</p>

### Motivation
How to train a deep learning model when I don't have hardware power to train the model, and even store all the tons of data that the model requires? Well... I go to the cloud.

### Objective
This repository examplifies how to traing a job on **GCP** when the input data is too big to be able to store it locally; in other words, the input data is also **in the cloud**, more specifically in a GCS bucket.

The model itself consist of a **Convolutional Neural Network**, build and trained purely on **TensorFlow**, in order to classify images from cats and dogs. The objective of this repository it's not to enter into the details of deep classifier models, but to show how to encapsulate a working package and send a training job to the Google compute engine.

### Training on the Cloud
Run the following command:

`source train_cloudml.sh`

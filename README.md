# Cats vs Dogs

<p align="center">
<img src="catdog.gif" width="300" ></a>
</p>

### Motivation
How to train a deep learning model when I don't have enough hardware power to train the model, and even store all the tons of data that the model requires? 

Well... I go to the cloud.

### Objective
This repository examplifies how to traing a job on **GCP** when the input data is too big to be able to store it locally; in other words, the input data is also **in the cloud**, more specifically in a GCS bucket.

The model itself consist of a **Convolutional Neural Network**, build and trained purely on **TensorFlow**, in order to classify images from cats and dogs. The objective of this repository it's not to enter into the details of deep classifier models, but to show how to encapsulate a working package and send a training job to the Google Compute Engine.

### Input Data
As an input data I used the well known Kaggle Cats Vs. Dogs dataset. Please update the dataset into your Cloud Storage Bucket following this hierarchy of folders:

    |- <BUCKET_NAME>
          └── cats_vs_dogs
              └── trained_models
              └── data
                  └── train
                      └── cats
                      └── dogs
                  └── validation
                      └── cats
                      └── dogs
                      
Where the trainining and validation JPEG images goes inside the respective `cats` or `dogs` folder.
                  
### Training on the Cloud
After making sure that you have installed the lastest version of the Google Cloud SDK, just run the following command:

`source train_cloudml.sh <BUCKET_NAME>`

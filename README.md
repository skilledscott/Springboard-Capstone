# UC San Diego Extended Studies - Machine Learning Capstone Project

### Author: Scott Gibson
### Course: September 2022 - March 2023

---

## Description

This repository will hold the work and analysis of my Capstone project for this course.

The goal of this project will be to experiment with different pretrained models and
learn more about the field of Machine Learning and Computer Vision. Specifically,
my focus will be on Object Detection.

While this repository will hold documents and notebooks describing my research and 
experimentation throughout this project, the final result will be a Web Application built
with React and Flask. This will be done in a different repository for the sake of clarity,
and that can be viewed here: https://github.com/Scott-Gibson-Coding/Springboard-Capstone-Production

---

## Repository Structure

### Documents

- Throughout the six month course, several writing assignments were given to help
  outline the thought process behind the project, and how progress is going. This
  varies from explaining the problem and dataset, or describing the intended methods
  of production and deployment.

### dataset_small

- Here is a small subset of images used during initial experimentation and eventually
  used when testing my web app in production. This folder has changed several times,
  originally around 200 images (a small subset of the original dataset from Roboflow)
  was kept in this folder, but since I wanted to push some to git, and many of the
  images I thought were less interesting, or having too many overlapping objects,
  so I removed enough to have a little over a dozen images.

### dataset_transform

- The notebooks in this folder used pandas and shutil to transform or reduce the dataset
  for different purposes.
- retinanet_to_gs_dataset_transform.ipynb was used to convert the _annotations.csv file
  I had to a format understood by Google Cloud, which I was using to make a benchmark
  at the time. This file also reduces the original dataset size, as I thought it would
  be too expensive to upload the original thousands of images to google cloud.
- dataset-transform.ipynb is a considerably simpler file, focusing on reducing the
  dataset size, and in my exploration phase I was also focusing entirely on bounding
  boxes that labeled cars. So anything related to pedestrians or bicycles were removed.

### models_exploration

- This folder contains a few images and some annotations that were used to demonstrate
  some OpenCV operations in the Jupyter Notebook file. This notebook used PyTorch
  to demonstrate image classification, and later object detection. I also experimented
  a bit with getting image predictions, and then displaying the mAP from the predetermined
  labels, thus allowing some insight into the performance of the model.

### prototype

- This folder split up the code from models_exploration into Python files that could be
  interacted with via the console. I tried to keep this fairly bare-bones, and more
  of a proof of concept. After all, since my final product would be a web app, making
  a console based application too fancy would be taking time away from the end result,
  which I didn't deem necessary at the time.

---

## Datasets

Since the datasets used in this project will be composed of thousands of images, I will not be posting them to GitHub. There are a few files in the dataset_small/ directory
that give information about the original dataset and its source, so please check that
out if you're interested.

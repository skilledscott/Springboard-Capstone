# Prototype

## Description

This prototype is a simple example of an object detection
workflow using PyTorch.

The dataset directory pointed to by the model.py constant 
DATASET_PATH has 416x416 .jpg images, and a .csv annotation file 
called _annotations.csv that has labels and bounding boxes for
each image. These will be trusted as the true labels, and referenced
for comparing the model predictions against.

An example dataset that you can use is provided, called dataset_small.
Change the chosen image by altering the IMAGE_NAME constant in model.py.

## Usage

Run the code by calling in the console, 'python model.py'. It will
display the mean Average Precision, and display the original image
with the model predictions drawn onto it.

At the moment, the user must interect directly with model.py
in order to change the inputs to the prototype.

At the top of model.py, there is a series of constants that can be
changed by the user.

- DATASET_PATH - relative path from model.py to the dataset
                 folder
- IMAGE_NAME - name of the image file to input to the 
               prototype
- DEVICE - set to either 'cpu' or 'gpu' to change between
           running model on cpu power, or gpu power.

## Notes

- Running a single image through the model on the gpu isn't
  considerably faster than cpu at this point. Running different
  trials has shown that the first batch of images run through the
  gpu is always slower, and every batch afterwards sees a considerable
  speedup. So gpu setting will mostly only be useful later if batches of
  images are run through the model.
- Testing has not yet been made on most of the pipeline. So it is likely
  that the user may run into errors if images aren't 416x416, amongst other
  points of weakness in the code.

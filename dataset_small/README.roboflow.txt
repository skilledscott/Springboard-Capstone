
Car-1 - v1 2022-04-04 2:15pm
==============================

This dataset was exported via roboflow.com on October 4, 2022 at 8:37 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 9676 images.
Car are annotated in retinanet format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)
* Grayscale (CRT phosphor)
* Auto-contrast via adaptive equalization

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Randomly crop between 0 and 20 percent of the image
* Random exposure adjustment of between -25 and +25 percent
* Random Gaussian blur of between 0 and 10 pixels
* Salt and pepper noise was applied to 5 percent of pixels



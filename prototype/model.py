import torch
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

import pandas as pd
import numpy as np

from time import time

from utils import (load_labels_df, 
    get_true_labels, 
    get_mean_average_precision,
    print_metric,
    draw_preds
)

# loads the PyTorch Faster_RCNN model
# returns model, preprocessing transform, and weight categories
def load_model():
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()

    preprocess = weights.transforms()
    weights_categories = weights.meta["categories"]

    return model, preprocess, weights_categories


# device can be ['cpu', 'cuda']
def get_batch_preds(img_names, ann_df, device='cpu', verbose=True):
    if verbose:
        print('Running model on: [{}] - {} images'.format(device, len(img_names)))
    
    pipeline_start_time = time()

    # check to make sure PyTorch sees the gpu before loading model and 
    # batch of images
    if not torch.cuda.is_available():
        print('\tNo device detected, running on cpu')
        device = 'cpu'

    # load model
    model, preprocess, weights_categories = load_model()
    model.to(device)

    if verbose:
        print('\tModel loaded         - {} seconds'.format(time() - pipeline_start_time))

    # run images through model in batches of 5 at a time
    batch_size = 5
    total = len(img_names)
    left = 0
    preds = []
    prediction_start_time = time()
    while left < total:
        # load batch of images from dataset
        right = left + batch_size
        imgs = [read_image(DATASET_PATH + name) for name in img_names[left:right]]
        if device == 'cuda':
            imgs = [img.cuda() for img in imgs]

        # create batch using preprocess on input batch of images
        batch = [preprocess(img) for img in imgs]

        # get predictions for the batch of images
        with torch.no_grad():
            batch_preds = model(batch)

        # append the batch preds to the preds array
        preds += batch_preds

        left += batch_size

    if verbose:
        print('\tPredictions finished - {} seconds'.format(time() - prediction_start_time))
        print('\n\tTotal pipeline       - {} seconds'.format(time() - pipeline_start_time))

    # put all output tensors on cpu for ease of scoring later
    if device != 'cpu':
        for i in range(len(preds)):
            preds[i]['boxes'] = preds[i]['boxes'].to('cpu')
            preds[i]['labels'] = preds[i]['labels'].to('cpu')
            preds[i]['scores'] = preds[i]['scores'].to('cpu')

    return preds


# set constant dataset path for input images
DATASET_PATH = '../dataset_small/test/'

# load annotations for dataset
ann_df = load_labels_df(DATASET_PATH)

# create a list of input images and run model to get
# batch predictions
# currently gets one random image, and runs preds on that
n_images = 1
img_nums = np.random.randint(1, 201, n_images)
img_names = ['img{}.jpg'.format(num) for num in img_nums]
preds = get_batch_preds(img_names, ann_df, device='cuda', verbose=True)

# get map
metric = get_mean_average_precision(img_names, ann_df, preds)

# print metric evaluation
print_metric(metric)

# draw preds and display/save to file (display for now)
draw_preds(preds)

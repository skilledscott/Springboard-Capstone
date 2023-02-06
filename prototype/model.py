import torch
from torchvision.io import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

import numpy as np
from time import time
from utils import (load_labels_df,
    filter_preds,
    get_true_labels,
    convert_labels_to_tensor,
    get_mean_average_precision,
    print_metric,
    draw_preds
)

# Constants
DATASET_PATH = '../dataset_small/'
IMAGE_NAME = 'img74.jpg'
DEVICE = 'gpu'


def load_model():
    '''
    load_model: Loads the faster_rcnn model from PyTorch with default
                weights.
                
                The preprocessing returned is to be run on images before
                inputting them into the model for prediction.

                The meta_categories returned are an array of categories,
                so if the model returns a label of 3, that's referring to
                meta_categories[3] i.e. 'car'.
    
    Returns model, preprocess, meta_categories
    '''

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()

    preprocess = weights.transforms()
    meta_categories = weights.meta["categories"]

    return model, preprocess, meta_categories


def get_preds(model, preprocess, meta_categories, img_names, device='cpu'):
    print('Running model on: [{}] - {} images'.format(device, len(img_names)))
    
    prediction_start_time = time()

    # run images through model in batches of 5 at a time
    batch_size = 5
    left = 0
    preds = []
    while left < len(img_names):
        # load batch of images from dataset
        right = left + batch_size
        imgs = [read_image(DATASET_PATH + name) for name in img_names[left:right]]

        # preprocess images and load them to gpu if necessary
        batch = [preprocess(img) for img in imgs]
        if device == 'cuda':
            batch = [img.cuda() for img in batch]

        # get predictions for the batch of images
        with torch.no_grad():
            batch_preds = model(batch)

        # append the batch preds to the preds array
        preds += batch_preds

        left += batch_size

    # filter preds and keep only the 'car' labels
    filtered_preds = filter_preds(preds, meta_categories, device)

    print('\tPredictions finished - {} seconds'.format(time() - prediction_start_time))
    
    return filtered_preds


def get_targets(img_names, ann_df, meta_categories, device):
    targets = []
    for name in img_names:
        labels, boxes = get_true_labels(name, ann_df)
        labels = convert_labels_to_tensor(labels, meta_categories)
        targets.append({
            'boxes': boxes if device == 'cpu' else boxes.cuda(),
            'labels': labels if device == 'cpu' else labels.cuda()
        })
    
    return targets


if __name__ == '__main__':
    # Make sure that cuda is available if running on gpu
    device = DEVICE
    if DEVICE == 'gpu':
        assert torch.cuda.is_available(), \
            'Error: DEVICE set to gpu, no gpu available. Change DEVICE to cpu in constants.'
        device = 'cuda'

    # load model, preprocess, and meta_categories
    model, preprocess, meta_categories = load_model()
    model.to(device)

    # load annotations for dataset
    ann_df = load_labels_df(DATASET_PATH)

    # create an array of images to read from the dataset
    img_names = [IMAGE_NAME]

    # get predictions from the model on the array of img_names
    preds = get_preds(model, preprocess, meta_categories, img_names, device)

    # get target, true labels from the annotations df
    targets = get_targets(img_names, ann_df, meta_categories, device)

    # get map comparing preds to true labels
    metric = get_mean_average_precision(preds, targets)

    # print metric evaluation
    print_metric(metric)

    # draw preds and display/save to file (display for now)
    draw_preds(DATASET_PATH, img_names, preds)

    print('\nDone')

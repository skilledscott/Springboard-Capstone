import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

# reads in the dataset _annotations.csv file
# returns a pandas DataFrame
# MOVE TO UTILS
def load_labels_df(path):
    '''
    load_labels_df: 
    '''
    # read in _annotations.csv file with bboxes and labels into ann_df
    names = ['img_name', 'x1', 'y1', 'x2', 'y2', 'label']
    ann_df = pd.read_csv(path + '_annotations.csv', header=None, names=names)
    return ann_df

# get's the true labels of an image from the annotations df
# returns labels, bounding box coordinates
# MOVE TO UTILS
def get_true_labels(img_name, ann_df):
    img_df = ann_df[ann_df['img_name'] == img_name]
    boxes_df = img_df[['x1', 'y1', 'x2', 'y2']]
    
    labels = list(img_df['label'])
    labels = torch.tensor([3 for _ in labels])
    boxes = torch.tensor(boxes_df.to_numpy())

    return labels, boxes


# only compatible with cuda tensors at the moment
# returns the mean average precision metric based on the annotations df and
# the model predictions
# returns the computed metric
# MOVE TO UTILS
def get_mean_average_precision(img_names, ann_df, preds):
    targets = [] # one dict per image

    for img_name in img_names:
        labels, boxes = get_true_labels(img_name, ann_df)
        targets.append({
            'boxes': boxes,
            'labels': labels
        })

    metric = MeanAveragePrecision()
    metric.update(preds, targets)
    return metric


def print_metric(metric):
    print('\nMean Average Precision:')
    pprint(metric.compute())
    return metric


# to be implemented...
def draw_preds(preds):
    print('\nDrawing bounding boxes...')
    print(preds[0])

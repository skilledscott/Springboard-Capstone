import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
from PIL import Image


def load_labels_df(path):
    '''
    load_labels_df: Uses pandas to read the csv at the location, 
                    '<path>/_annotations.csv', and returns the read 
                    DataFrame.
    
    :param path: str, image dataset path where .csv file is stored

    Returns ann_df -> pd.DataFrame
    '''
    # read in _annotations.csv file with bboxes and labels into ann_df
    names = ['img_name', 'x1', 'y1', 'x2', 'y2', 'label']
    ann_df = pd.read_csv(path + '_annotations.csv', header=None, names=names)
    return ann_df


def get_true_labels(img_name, ann_df):
    '''
    get_true_labels: Filters ann_df for all rows for the given img_name.
                     Returns a tensor of labels, and a tensor of bounding boxes.
    
    :param img_name: str
    :param ann_df: pd.DataFrame

    Returns labels -> list[str], boxes -> torch.tensor(list[list[float]])
    '''
    img_df = ann_df[ann_df['img_name'] == img_name]
    boxes_df = img_df[['x1', 'y1', 'x2', 'y2']]
    
    labels = list(img_df['label'])
    boxes = torch.tensor(boxes_df.to_numpy())

    return labels, boxes

def convert_labels_to_tensor(labels, meta_categories):
    '''
    convert_labels_to_tensor: Convert a list of strings to their integer
                              index location in meta_categories. Will return a
                              tensor of integers stored on the cpu.

    :param labels: list[str], bounding box labels, like 'car', or 'bike'
    :param meta_categories: list[str], each str corresponds to an output label.
                       if meta_categories[3] = 'car' -> 'car' = 3.
    
    Returns labels -> torch.tensor(list[int])
    '''

    int_labels = []
    for label in labels:
        for index in range(len(meta_categories)):
            if meta_categories[index] == label:
                int_labels.append(index)
                break
            
    return torch.tensor(int_labels)


def filter_preds(preds, meta_categories, device):
    # filter preds and keep only the 'car' labels
    car_label = [i for i in range(len(meta_categories)) if
                    meta_categories[i] == 'car'][0]
    filtered_preds = []
    for pred in preds:
        keep_indeces = torch.tensor([i for i in range(len(pred['labels']))
                                        if pred['labels'][i] == car_label]).to(device)
        filtered_pred = {
            'boxes': torch.index_select(pred['boxes'], 0, keep_indeces),
            'labels': torch.index_select(pred['labels'], 0, keep_indeces),
            'scores': torch.index_select(pred['scores'], 0, keep_indeces)
        }
        filtered_preds.append(filtered_pred)
    
    return filtered_preds


def get_mean_average_precision(preds, targets):
    '''
    get_mean_average_precision: Inits and updates a PyTorch MeanAveragePrecision
                                metric and returns that metric to the user. The
                                metric can be visualized by calling metric.compute().
                                Or by calling print_metric from utils.py.
    
    :param img_names: list[str], list of image names, should be 1-to-1 with preds
    :param ann_df: pd.DataFrame, df of true labels for images
    :param preds: list[dict['boxes', 'labels', 'scores']], each image from img_names
                  corresponds to an entry in this list

    Returns metric -> MeanAveragePrecision object
    '''

    metric = MeanAveragePrecision()
    metric.update(preds, targets)
    return metric


# print_metric: Uses pprint and metric.compute() to display metric information
# to the console.
def print_metric(metric):
    print('\nMean Average Precision:')
    pprint(metric.compute())


def draw_preds(dataset_path, img_names, preds):
    '''
    draw_preds: Uses plt.imshow to display the bounding boxes and labels from
                preds on the first input image.
    
    :param dataset_path: str, path to the dataset of images
    :param img_names: list[str], each image should match one-to-one with the preds
                      list
    :param preds: list[dict['boxes', 'labels', 'scores']]
    '''

    print('\nDrawing bounding boxes...')

    boxes = preds[0]['boxes']
    labels = ['car' for _ in preds[0]['labels']]

    img = read_image(dataset_path + img_names[0])
    boxes_img = draw_bounding_boxes(image=img, 
                                    boxes=boxes.to('cpu'), 
                                    labels=labels,
                                    colors='red',
                                    width=4,
                                    font='arial',
                                    font_size=24)
    plt.imshow(to_pil_image(boxes_img))
    plt.show()

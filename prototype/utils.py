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
    load_labels_df: Uses pandas to read '<path>/_annotations.csv'
    
    :param path: str, image dataset path where the .csv file is stored

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
    
    :param img_name: str, name used to filter ann_df
    :param ann_df: pd.DataFrame, dataset boxes and labels

    Returns labels -> list[str], boxes -> torch.tensor(list[list[float]])
    '''
    img_df = ann_df[ann_df['img_name'] == img_name]
    boxes_df = img_df[['x1', 'y1', 'x2', 'y2']]
    
    labels = list(img_df['label'])
    boxes = torch.tensor(boxes_df.to_numpy())

    return labels, boxes

def convert_labels_to_tensor(labels, meta_categories):
    '''
    convert_labels_to_tensor: Convert a list of strings to the weights representation.
                              Will return a tensor of integers.

    :param labels: list[str], bounding box labels, like 'car', or 'bike'
    :param meta_categories: list[str], indexed categories from weights.
    
    Returns labels -> torch.tensor(list[int])

    Labels returned will be a tensor stored on the cpu, it needs to be moved to the gpu
    later if necessary.
    '''

    int_labels = []
    for label in labels:
        for index in range(len(meta_categories)):
            if meta_categories[index] == label:
                int_labels.append(index)
                break
            
    return torch.tensor(int_labels)


def filter_preds(preds, meta_categories, device):
    '''
    filter_preds: Filters the preds to only return the 'car' bounding boxes.

    :param preds: list[dict{boxes, labels, scores}], predictions to filter
    :param meta_categories: list[str], list of weight categories
    :param device: str in ['cpu', 'cuda'], use cuda to return tensors on the gpu

    Returns filtered_preds -> list[dict{boxes, labels, scores}]
    '''
    
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
    
    :param preds: list[dict{boxes, labels, scores}], predictions from model
    :param targets: list[dict{boxes, labels}], true labels from dataset .csv file

    Returns: metric -> MeanAveragePrecision object

    Call print_metric with the returned metric to print it to the console.
    '''

    metric = MeanAveragePrecision()
    metric.update(preds, targets)
    return metric


def print_metric(metric):
    '''
    print_metric: Prints the metric to the console using pprint.

    :param metric: MeanAveragePrecision object
    '''

    print('\nMean Average Precision:')
    pprint(metric.compute())


def draw_preds(dataset_path, img_names, preds):
    '''
    draw_preds: Uses plt.imshow to display the bounding boxes and labels from
                preds on the first input image.
    
    :param dataset_path: str, path to the dataset of images
    :param img_names: list[str]
    :param preds: list[dict['boxes', 'labels', 'scores']]

    Currently only draws the first image from img_names.
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

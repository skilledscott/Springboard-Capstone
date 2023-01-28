import torch
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from time import time

DATASET_PATH = '../dataset_small/test/'


def load_model():
    # load model with pretrained weights and threshold
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()

    preprocess = weights.transforms()

    weights_categories = weights.meta["categories"]

    return model, preprocess, weights_categories


def load_labels_df():
    # read in _annotations.csv file with bboxes and labels into ann_df
    names = ['img_name', 'x1', 'y1', 'x2', 'y2', 'label']
    ann_df = pd.read_csv(DATASET_PATH + '_annotations.csv', header=None, names=names)
    return ann_df


def get_true_labels(ann_df, img_name):
    pass

def run_pipeline(img_names, device='cpu'):
    pipeline_start_time = time()

    # uncomment this next line to enable gpu if available
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # load model
    model, preprocess, weights_categories = load_model()
    model.to(device)

    start_time = time()

    batch_size = 5
    total = len(img_names)
    left = 0
    preds = []
    while True:
        if left >= total:
            break
        if left + batch_size >= total:
            right = total
        else:
            right = left + batch_size

        # load batch of images from dataset
        # imgs = [read_image(DATASET_PATH + name).cuda() for name in img_names[left:right]]
        imgs = [read_image(DATASET_PATH + name).cuda() if device != 'cpu' \
                else read_image(DATASET_PATH + name) for name in img_names[left:right]]

        # create batch using preprocess on input batch of images
        batch = [preprocess(img) for img in imgs]

        # get predictions for the batch of images
        with torch.no_grad():
            batch_preds = model(batch)

        # append the batch preds to the preds array
        preds += batch_preds
        del(batch_preds)

        left += batch_size
    
    pred_time = time() - start_time
    pipeline_total_time = time() - pipeline_start_time
    return preds, pred_time, pipeline_total_time


# gets an average of 3 runtimes
def test_pipeline_runtime(n_images, device='cpu'):
    pred_times = []
    pipeline_total_times = []
    for _ in range(3):
        img_nums = np.random.randint(1, 201, n_images)
        img_names = ['img{}.jpg'.format(num) for num in img_nums]

        # run pipeline on img
        preds, pred_time, pipeline_total_time = run_pipeline(img_names, device)
        pred_times.append(pred_time)
        pipeline_total_times.append(pipeline_total_time)

    print('''Model average of 3 for {} images.
        prediction time:     {} seconds
        pipeline total time: {} seconds
        '''.format(n_images,
                   sum(pred_times) / 3,
                   sum(pipeline_total_times) / 3))

# load labels into df
# ann_df = load_labels_df()

test_pipeline_runtime(1, device='cpu')
# test_pipeline_runtime(1, device='cuda:0')

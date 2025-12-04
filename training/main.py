import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import pandas as pd
import PIL
from PIL import Image
import os
import tqdm
import hashlib
import matplotlib.pyplot as plt
import cv2
import itertools
from math import floor, ceil
import albumentations as A
import copy

from load import *
import corners
#import similar
import clip
import properties

# test set split using random_state 1337
#SIMILAR_MODEL_PATH = ""

PROPERTY_MODEL_PATH = "models/convnet-attn_1308e46f_epoch29_time1764865520"
# Epoch 29: Train err: 0.0026058631921824105, Train loss: 0.011797186569310725 |Validation err: 0.0078125, Validation loss: 0.035031265944780
# Test err: 0.017578125, Test loss: 0.04996311323156988

CORNERS_MODEL_PATH = "models/cornerNet-features_7472f19c_epoch50_time1764867421"
#Epoch 50: Train err: 0.1472312703583062, Train loss: 0.00016075088115030667 |Validation err: 0.126953125, Validation loss: 0.00018415241720504127
#Test err: 0.1640625, Test loss: 0.00018083257418766152

#similarnet = similar.net_type().to(device)
#similarnet.load_state_dict(torch.load(SIMILAR_MODEL_PATH)["net"])
propertynet = properties.net_type().to(device)
propertynet.load_state_dict(torch.load(PROPERTY_MODEL_PATH)["net"])
cornernet = corners.net_type().to(device)
cornernet.load_state_dict(torch.load(CORNERS_MODEL_PATH)["net"])
#similarnet.eval()
propertynet.eval()
cornernet.eval()


def final_model(img_path):
    #print(img)
    img = get_image(img_path)
    corners_heatmap = cornernet(torch.tensor(img).to(device).unsqueeze(0))[0] * (1024//XMAX)
    tocrop = corners.read_full_heatmap(corners_heatmap.detach().cpu().numpy())

    img = get_image(img_path, fullsize=True)

    img = np_to_img(img)
    img = transform_4point(img, tocrop)
    ret_img = img
    img = img_to_np(img)

    color = torch.argmax(propertynet(torch.tensor(img).to(device).unsqueeze(0))[0])
    color = propmap[int(color.detach().cpu())]

    ordered = clip.query(img)

    for cardid in ordered:
        if cards[cardid][PROP] == color:
            return cardid, ret_img

for i in tqdm.tqdm(image_data, total=len(image_data)):
    img_path = os.path.join(image_path, i['filename'])
    #print(final_model(img_path), i['card_id'])

TEST = 0
if TEST:
    success = 0
    count = 0
    #for i in train:
    for i in val:
    #for i in test:
        img_path = os.path.join(image_path, i['filename'])

        count += 1
        if final_model(img_path)[0] == i['card_id']:
            success += 1

    print(success/count)

TEST_REAL = 1
if TEST_REAL:
    #for i in Path("/home/john/docs/schoolwork/APS360 Deep Learning/real_data/cropped").iterdir():
    for i in Path("/home/john/docs/schoolwork/APS360 Deep Learning/real_data/new_batch/").iterdir():
        img_path = i.resolve()
        pred, ret_img = final_model(img_path)
        print(pred)

        f, subplot = plt.subplots(1,2)
        f.suptitle(pred, size=36)
        subplot[0].imshow(ret_img)
        subplot[1].imshow(np_to_img(get_image(img_path)))
        plt.show()

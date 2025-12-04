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
from torchvision import transforms
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
import albumentations as A
import random
import scipy.special
from scipy.optimize import linear_sum_assignment

from load import *

EPOCHS = 50

# ====================================== problem definition

def get_transform(epoch):
    strength = epoch/EPOCHS
    #strength = (epoch/EPOCHS)**2  # using x^2 as the strength regiment to give it a slower start
    #strength = 1

    torch_transform = transforms.Compose([
        transforms.ColorJitter(
            brightness=0.1+0.4*strength,
            contrast=0.1+0.4*strength,
            saturation=0.1+0.4*strength,
            hue=0.01+0.08*strength
        ),

        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=round_to_odd(1+2*strength), sigma=(0.1, 2.0))],
            p=strength
        ),

        ACoarseDropout(
            num_holes_range = (1,1+round(4*strength)),
            hole_height_range = (1,YMAX//4),
            hole_width_range = (1,XMAX//4),
            fill = "random",
            p = 0.5 + 0.5*strength,
        ),
    ])

    return torch_transform


class FeatureCornerDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        if ind in self.cache:
            return self.cache[ind]
        else:
            self.cache[ind] = self.getitem(ind)
            return self.cache[ind]

    def getitem(self, ind):
        datapoint = self.dataset[ind]

        sigma = round(XMAX / 64)  # 2 when xmax is 128

        heatmaps = []
        coords = get_coords_from(datapoint)
        for x, y in coords:
            xx, yy = np.meshgrid(np.arange(XMAX), np.arange(YMAX))
            heatmap = np.exp(-((xx-x)**2 + (yy-y)**2)/(2*sigma**2))
            heatmap = heatmap / heatmap.max()
            #heatmap = scipy.special.log_softmax(heatmap, axis=1)
            #heatmap = scipy.special.softmax(heatmap)
            #print(heatmap.max())
            heatmaps.append(np.array(heatmap, dtype="float32"))

        sumheatmap = sum(np.array(heatmaps))
        heatmaps = np.array([sumheatmap])  # 1 feature with all 4 corners
        #heatmaps = np.array(heatmaps)

        img_path = os.path.join(image_path, datapoint['filename'])
        img = get_image(img_path)

        return img, heatmaps


train_dataset = FeatureCornerDataset(train)
val_dataset = FeatureCornerDataset(val)
test_dataset = FeatureCornerDataset(test)




# ============================================== models


class CornerNet_Features(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet34(weights="IMAGENET1K_V1")

        # full encoder part of the backbone up to layer4
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.up4 = self.upsample(512, 256)
        self.up3 = self.upsample(256, 128)
        self.up2 = self.upsample(128, 64)
        self.up1 = self.upsample(64, 32)
        self.up0 = self.upsample(32, 32)
        self.final = nn.Conv2d(32, 1, kernel_size=1)

        self.name = "cornerNet-features"

    def upsample(self, dimin, dimout):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(dimin, dimout, kernel_size=3, padding=1),
            nn.BatchNorm2d(dimout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)

        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.up0(x)

        x = self.final(x)
        x = torch.sigmoid(x)
        return x


# ==================================== heatmap helper functions

#def read_heatmap(heatmap):
#    # unravel_index prolly changes the order of some things... I'm just inverting the index but if H and W are different there might be a bug
#    return np.array([np.unravel_index(np.argmax(i), i.shape)[::-1] for i in heatmap], copy=True)

def read_full_heatmap(heatmap):
    heatmap = heatmap[0]
    #heatmap = np.maximum(0, heatmap-0.5) * 2  # remove anything lower than 0.5

    #smoothed = heatmap
    smoothed = gaussian_filter(heatmap, sigma=XMAX//100)  # heatmap hyperparameter
    #smoothed = gaussian_filter(heatmap, sigma=XMAX//20)  # heatmap hyperparameter
    #print(np.max(smoothed))
    for i in range(70,1,-1):
        points = peak_local_max(smoothed, min_distance=XMAX//30, num_peaks=4, threshold_abs=np.max(smoothed)*i/100)  # heatmap hyperparameter
        if len(points) == 4:
            break
    #points = peak_local_max(smoothed, min_distance=XMAX//16, num_peaks=4)  # heatmap hyperparameter

    corners = np.array([
            [XMAX-1, 0],
            [0,0],
            [XMAX-1, YMAX-1],
            [0, YMAX-1],
        ])
    
    if len(points) != 4:
        return corners

    #points are in y, x
    points[:, 0], points[:, 1] = points[:, 1], points[:, 0].copy()

    cost = np.linalg.norm(points[:,None,:] - corners[None,:,:], axis=2)  # distances from points to corners
    row_ind, col_ind = linear_sum_assignment(cost)  # linear sum assignment: minimizes total distance

    sorted_pts = np.zeros_like(points)
    sorted_pts[col_ind] = points[row_ind]

    return sorted_pts

def read_heatmap_torch(heatmap):
    return torch.tensor(read_full_heatmap(heatmap.detach().cpu()))

read_heatmap = read_full_heatmap


ERROR_RAD = floor(XMAX/15) # further than this radius is an error
def is_error(outputs, labels):
    toret = []
    for output_item, label_item in zip(outputs, labels):
        euclidian_dists = torch.sum((read_heatmap_torch(label_item)-read_heatmap_torch(output_item))**2, dim=1)
        toret.append(torch.any(euclidian_dists > ERROR_RAD**2))
    return torch.stack(toret)


# ===================================== settings
# continue training from this
train_from = None
#train_from = "/home/john/docs/schoolwork/APS360 Deep Learning/training/models/cornerNet-features_7472f19c_epoch12_time1764821636"
TRAIN = 0
TEST = 0
TEST_REAL = 0
#default_test_path = "models/cornerNet-features_e3f826b0_epoch50_time1764870136"
default_test_path = "models/cornerNet-features_7472f19c_epoch50_time1764867421"
#net_type = CornerNet
net_type = CornerNet_Features
#net_type = CornerHeatmap1C
VISUALIZE = 0
# EPOCHS


# ===================================== training
net = net_type()
net = net.to(device)

#criterion = nn.KLDivLoss(reduction='batchmean')
#criterion = nn.BCEWithLogitsLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

#train_loader = DataLoader(FeatureCornerDataset(train[:64]), batch_size=64, shuffle=True)   # overfit to small dataset
#train_loader = DataLoader(FeatureCornerDataset(train[:16]), batch_size=16, shuffle=True)   # overfit to small dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)
model_path = None


if TRAIN:
    #stats = train_net(net, batch_size=4, learning_rate=1e-3, num_epochs=500, train_max=FeatureCornerDataset(train[:48]))
    #stats = train_net(net, batch_size=64, learning_rate=0.002, num_epochs=30)
    trainer = train_net(net, criterion, optimizer, train_loader, is_error, get_transform, train_from)

    stats = []
    model_path = ""
    for epoch, train_err, train_loss, model_path in trainer:
        val_err, val_loss = evaluate(net, val_loader, criterion, is_error, get_transform(epoch))
        stats.append([epoch, train_err, train_loss, val_err, val_loss])

        if VISUALIZE != 0 and epoch >= VISUALIZE - 1:
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                inputs = get_transform(epoch)(inputs)
                labels = labels.to(device)
                outputs = net(inputs)

                points = read_full_heatmap(outputs[0].detach().cpu())
                show_points(np_to_img(inputs[0].detach().cpu()), points)

                num_layers = len(outputs.detach().cpu()[0])
                for j in range(num_layers):
                    sns.heatmap(outputs.detach().cpu()[0][j])
                    plt.show()
                    sns.heatmap(labels.detach().cpu()[0][j])
                    plt.show()

                # display outputs[0], inputs[0], labels[0]
                break

        print(("Epoch {}: Train err: {}, Train loss: {} |"+
               "Validation err: {}, Validation loss: {}").format(
                   epoch,
                   train_err,
                   train_loss,
                   val_err,
                   val_loss))

        if epoch == EPOCHS:
            break

    #print("Model at:", model_path)
    plot_training_curve(stats, model_path)



# ======================================== testing

if TEST or TEST_REAL:
    net = net_type()
    net = net.to(device)

    if model_path is None:
        model_path = default_test_path # change this whenever you don't train
    state = torch.load(model_path)["net"]
    net.load_state_dict(state)


if TEST:
    test_err, test_loss = evaluate(net, test_loader, criterion, is_error, get_transform(EPOCHS))
    print(f"Test err: {test_err}, Test loss: {test_loss}")


if TEST_REAL:
    for datapoint in real_image_data:
        img = get_image(os.path.join(real_image_path, datapoint['filename']))

        img_batched_torch = torch.from_numpy(np.array([img])).to(device)

        heatmap = np.array(net(img_batched_torch)[0].detach().cpu())

        sns.heatmap(heatmap[0])
        plt.show()

        points = read_heatmap(heatmap)
        img = np_to_img(img)

        show_points(img, points)
        plt.imshow(img)
        plt.show()
        plt.imshow(transform_4point(img, points))
        plt.show()

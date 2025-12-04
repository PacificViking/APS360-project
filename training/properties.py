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
import cv2
import copy
import matplotlib.pyplot as plt
import random
from math import floor, ceil
import seaborn as sns

from load import *

EPOCHS = 30

# ======================================== problem definition

def get_transform(epoch):
    strength = (epoch/EPOCHS)
    #strength = (epoch/EPOCHS)**2  # using x^2 as the strength regiment to give it a slower start

    transform = transforms.Compose([
        transforms.ColorJitter(
            brightness=0.1+0.3*strength,
            contrast=0.1+0.3*strength,
            saturation=0.1+0.3*strength,
            hue=0.01+0.06*strength
        ),

        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=round_to_odd(1+2*strength), sigma=(0.1, 2.0))],
            p=strength
        ),

        ACoarseDropout(
            num_holes_range = (1,1+round(9*strength)),
            hole_height_range = (1,YMAX//16),
            hole_width_range = (1,XMAX//16),
            fill = "random",
            p = 0.3 + 0.7*strength,
        ),
    ])
    return transform


class PropertyDataset(Dataset):
    def __init__(self, dataset, prop):
        self.dataset = dataset
        self.prop = prop
        
        self.cache = {}

        #self.pbar = tqdm.tqdm(total=len(dataset))
        self.pbar = None


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        if ind not in self.cache:
            self.cache[ind] = self.getitem(ind)
            if self.pbar is not None:
                self.pbar.update(1)
                if len(self.cache) == len(self.dataset):
                    self.pbar.close()
        return self.cache[ind]

    def getitem(self, ind):
        datapoint = self.dataset[ind]
        #datapoint['coords']

        img = get_image(os.path.join(image_path, datapoint['filename']), fullsize=True)

        coords = get_coords_from(datapoint, fullsize=True)
        coords = noise_coords(coords, fullsize=True)

        img = np_to_img(img)
        img = transform_4point(img, coords, show=False)
        img = img_to_np(img)

        property_val = cards[datapoint['card_id']][self.prop]

        img = torch.tensor(img)
        prop = torch.tensor(propmap[property_val])

        return img, prop

train_dataset = PropertyDataset(train, PROP)
val_dataset = PropertyDataset(val, PROP)
test_dataset = PropertyDataset(test, PROP)


def is_error(outputs, labels):   # batchwise is_error
    return torch.argmax(outputs, dim=1) != labels



# ======================================= models

class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.modules.flatten.Flatten(2,3),
            nn.Softmax(dim=2)  # softmax2d does it over the channels too for some reason
            )

    def forward(self, x):
        weights = self.attention(x)

        x = torch.flatten(x, 2, 3)
        # so flatten both x and weights spatially, since I'm summing over them anyways

        x = x * weights
        x = x.sum(dim=2)  # weight, then sum spatial parts for a 64x1x1 channels

        #print(torch.mean(weights))

        return x


class ConvNet_Attn(nn.Module):
    def __init__(self, num_classes=len(propvalues)):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )

        self.attention = Attention(64)

        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes)
        )

        self.name = "convnet-attn"

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.classifier(x)
        return x




# ===================================== settings
# continue training from this
train_from = None
#train_from = "models/convnet-attn_7c633bed_epoch2_time1764740300"
TRAIN = 0
TEST = 0
TEST_REAL = 0
default_test_path = "models/convnet-attn_1308e46f_epoch27_time1764875243"
net_type = ConvNet_Attn
VISUALIZE = 0
# EPOCHS


# ===================================== training
net = net_type()
net = net.to(device)

fast_params = net.attention.parameters()
slow_params = itertools.chain(net.features.parameters(), net.classifier.parameters())

criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(net.parameters(), lr=1e-3)
optimizer = optim.Adam([
    {"params": slow_params, "lr": 1e-4},
    {"params": fast_params, "lr": 5e-4},
    ])

#train_loader = DataLoader(PropertyDataset(train[:64], PROP), batch_size=64, shuffle=True)   # overfit to small dataset
#train_loader = DataLoader(PropertyDataset(train[:16], PROP), batch_size=16, shuffle=True)   # overfit to small dataset
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)
model_path = None



if TRAIN:
    trainer = train_net(net, criterion, optimizer, train_loader, is_error, get_transform, train_from)
    #trainer = train_net(net, criterion, optimizer, train_loader, is_error, None, train_from)

    stats = []
    model_path = ""
    for epoch, train_err, train_loss, model_path in trainer:
        val_err, val_loss = evaluate(net, val_loader, criterion, is_error, get_transform(epoch))
        #val_err, val_loss = evaluate(net, train_loader, criterion, is_error, None)
        stats.append([epoch, train_err, train_loss, val_err, val_loss])

        if VISUALIZE:
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)

                show_image(np_to_img(inputs[0].detach().cpu()))
                print(labels[0])
                print(outputs[0])
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
        img = get_image(os.path.join(real_image_path, datapoint['filename']), fullsize=True)
        coords = get_coords_from(datapoint, fullsize=True)

        img = np_to_img(img)
        show_image(img)
        img = transform_4point(img, coords, show=False)
        img = img_to_np(img)

        img_batched_torch = torch.from_numpy(np.array([img])).to(device)
        print(propmap[int(torch.argmax(net(img_batched_torch).detach()).cpu())])

        show_image(np_to_img(img))


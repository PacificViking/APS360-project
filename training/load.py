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

XMAX = 256
YMAX = 256
#XMAX = 1024
#YMAX = 1024

fab_card_size = (768, 384)

device = torch.device("cuda")


# ==================================== card data

image_path = Path(os.path.expanduser("~/schoolwork/APS360 Deep Learning/data_generation/images")).resolve()
image_data_path = Path(os.path.expanduser("~/schoolwork/APS360 Deep Learning/data_generation/cards.json")).resolve()

card_images_path = Path(os.path.expanduser("~/schoolwork/APS360 Deep Learning/fabscrape/images")).resolve()
card_data_path = Path(os.path.expanduser("~/schoolwork/APS360 Deep Learning/fabscrape/cards.json")).resolve()

real_image_path = Path(os.path.expanduser("~/schoolwork/APS360 Deep Learning/real_data/cropped")).resolve()
real_image_data_path = Path(os.path.expanduser("~/schoolwork/APS360 Deep Learning/real_data/cards.json")).resolve()

with open(image_data_path, 'r') as file:
    image_data = json.load(file)

with open(real_image_data_path, 'r') as file:
    real_image_data = json.load(file)

with open(card_data_path.resolve(), 'r') as file:
    card_data = json.load(file)

image_data.sort(key=lambda x: x['card_id'])
train_val, test = train_test_split(image_data, test_size=0.2, shuffle=True, random_state=1337)
train, val = train_test_split(train_val, test_size=(0.2/0.8), shuffle=True, random_state=1337)

cards = {}
for i in card_data:
    cards[i['card_id']] = i





# ================================= image manipulation generics

def round_to_odd(x):
    if x % 2 == 0:
        return x + 1
    if floor(x) % 2 == 0:
        return ceil(x)
    else:
        return floor(x)

def show_image(img, title=None):
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()

def show_points(img, points):
    plt.imshow(img)
    plt.scatter(points[:,0], points[:,1], c='red')

    for i, (x, y) in enumerate(points):
        plt.text(x, y, ' '+str(i), color='red')

    plt.show()

img_cache = {}
def get_image(img_path, fullsize=False):
    if img_path not in img_cache:
        img = Image.open(img_path).convert("RGB")
        if not fullsize:
            img = transforms.Resize((XMAX, YMAX))(img)

        img = np.array(img)
        img = np.transpose(img, [2,0,1])
        img = np.array(img/255, dtype=np.float32)

        img_cache[img_path] = img

    return img_cache[img_path]

def get_coords_from(datapoint, fullsize=False):
    out = []
    for i in datapoint['coords']:
        if fullsize:
            out.append([round(i[0]), 1023-round(i[1])])
        else:
            out.append([round(i[0]*XMAX/1024), (YMAX-1)-round(i[1]*YMAX/1024)])
    return np.array(out)

def transform_4point(img, points, show=False):
    #https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    points = np.array(points, dtype="float32")
    maxx, maxy = (546, 762)
    dst = np.array([
        [maxx-1, 0],
        [0, 0],
        [maxx-1, maxy-1],
        [0, maxy-1],
        ], dtype="float32")

    M = cv2.getPerspectiveTransform(points, dst)
    warped = cv2.warpPerspective(img, M, (maxx, maxy))

    if show:
        plt.imshow(warped)
        plt.show()

    return warped


def np_to_img(img):
    return np.transpose(img, [1,2,0])
def img_to_np(img):
    return np.transpose(img, [2,0,1])
def np_to_pil(img):
    return Image.fromarray((np_to_img(img) * 255).astype('uint8'), 'RGB')

def sha256(x):
    m = hashlib.sha256()
    m.update(x.encode('utf-8'))
    return str(m.hexdigest())




# ==========================================  # properties preprocessing
PROP = "pitch"
#PROP = "defense"

propvalues = set()
for i in card_data:
    propvalues.add(i[PROP])
propvalues = list(propvalues)
propvalues.sort()  # deterministic order

print(PROP, propvalues)

propmap = {}
for i, j in enumerate(propvalues):
    assert not isinstance(j, int)
    propmap[i] = j
    propmap[j] = i






# ===========================================   error evaluation
def evaluate(net, loader, criterion, f_is_error, transform=None, triplet_loss=False):
    # this function is copied and lightly modified from lab2
    net.eval()
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    err = -1
    loss = -1
    for i, data in enumerate(loader):
        if triplet_loss:
            anchor, pos, neg = data
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            if transform is not None:
                anchor = transform(anchor)
                pos = transform(pos)
                neg = transform(neg)

            loss = criterion(net(anchor), net(pos), net(neg))
            corr = 0
            epoch_len = len(anchor)
        else:
            inputs, labels = data
            inputs = inputs.to(device)
            if transform is not None:
                inputs = transform(inputs)
            labels = labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            corr = f_is_error(outputs, labels).sum()
            epoch_len = len(labels)

        total_err += corr
        total_loss += loss.item()
        total_epoch += epoch_len
        err = float(total_err) / total_epoch
        loss = float(total_loss) / (i + 1)
    net.train()
    return err, loss

# ============================================  training utils

def plot_training_curve(stats, title):
    # this function is copied and lightly modified from lab2
    stats = np.transpose(np.array(stats))
    epoch, train_err, train_loss, val_err, val_loss = stats
    plt.title("Error (train has dropout, val doesn't)")
    plt.suptitle(title)
    plt.plot(epoch, train_err, label="Train")
    plt.plot(epoch, val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    plt.title("Loss (train has dropout, val doesn't)")
    plt.plot(epoch, train_loss, label="Train")
    plt.plot(epoch, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()


def train_net(net, criterion, optimizer, train_loader, f_is_error, transform_curriculum=None, checkpoint=None, triplet_loss=False):
    # this function is copied and lightly modified from lab3
    #torch.manual_seed(1000)

    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        epoch = 0

    while True:
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        i = -1
        for i, data in enumerate(train_loader):
            # Get the inputs

            if triplet_loss:
                anchor, pos, neg = data
                anchor = anchor.to(device)
                pos = pos.to(device)
                neg = neg.to(device)

                if transform_curriculum is not None:
                    anchor = transform_curriculum(epoch)(anchor)
                    pos = transform_curriculum(epoch)(pos)
                    neg = transform_curriculum(epoch)(neg)

                optimizer.zero_grad()
                loss = criterion(net(anchor), net(pos), net(neg))
                corr = 0
                epoch_len = len(anchor)
            else:
                inputs, labels = data
                inputs = inputs.to(device)

                if transform_curriculum is not None:
                    inputs = transform_curriculum(epoch)(inputs)

                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass, backward pass, and optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                corr = f_is_error(outputs, labels).sum()
                epoch_len = len(labels)

            loss.backward()
            optimizer.step()
            # Calculate the statistics
            total_train_err += corr
            total_train_loss += loss.item()
            total_epoch += epoch_len
        train_err = float(total_train_err) / total_epoch
        train_loss = float(total_train_loss) / (i+1)

        epoch += 1

        # Save the current model (checkpoint) to a file
        model_path = f"models/{net.name}_{sha256(str(net))[:8]}_epoch{epoch}_time{int(time.time())}"
        checkpoint = {
                "epoch": epoch,
                "net": net.state_dict(),
                "optimizer": optimizer.state_dict()
                }
        #torch.save(net.state_dict(), model_path)
        torch.save(checkpoint, model_path)

        print("Model at:", model_path)

        yield epoch, train_err, train_loss, model_path





# ===================== preprocessing utilities

class ACoarseDropout(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.transform = A.Compose([
            A.CoarseDropout(**kwargs),
            A.pytorch.ToTensorV2(),   # this does img->np transpose automatically
            ])

    def forward(self, img):
        numpied = img.cpu().numpy()
        if len(numpied.shape) == 4:
            return torch.stack([self.use_transform(i) for i in numpied])
        else:
            assert len(numpied.shape) == 3
            return self.use_transform(numpied)

    def use_transform(self, np_img):
        transformed = self.transform(image=np_to_img(np_img))["image"].to(device)
        return transformed


def noise_coords(coords, noise_scale=4, fullsize=False):
    if fullsize:
        xmax = 1024
        ymax = 1024
    else:
        xmax = XMAX
        ymax = YMAX
    if noise_scale is not None:
        #noise_x = np.int64(np.random.uniform(-XMAX/100*NOISE_SCALE, XMAX/100*NOISE_SCALE, 4))
        #noise_y = np.int64(np.random.uniform(-YMAX/100*NOISE_SCALE, YMAX/100*NOISE_SCALE, 4))
        noise_x = np.int64(np.random.uniform(0, xmax/100, 4)) * np.array([1, -1, 1, -1]) * noise_scale # only expand
        noise_y = np.int64(np.random.uniform(0, ymax/100, 4)) * np.array([-1, -1, 1, 1]) * noise_scale
        #noise_x = np.int64(np.random.uniform(-XMAX/100/5, XMAX/100, 4)) * np.array([1, -1, 1, -1]) * NOISE_SCALE  # expand, contract less
        #noise_y = np.int64(np.random.uniform(-YMAX/100/5, YMAX/100, 4)) * np.array([-1, -1, 1, 1]) * NOISE_SCALE
    else:
        noise_x = np.int64(np.zeros(4))
        noise_y = np.int64(np.zeros(4))
    noisy = coords.copy()
    noisy[:, 0] += noise_x
    noisy[:, 1] += noise_y
    noisy[:, 0] = np.clip(noisy[:, 0], 0, xmax - 1)
    noisy[:, 1] = np.clip(noisy[:, 1], 0, ymax - 1)
    return noisy

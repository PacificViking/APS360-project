from PIL import Image
import imagehash
from pathlib import Path
import os
import json
import tqdm
import base64
import pickle
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import sys

PHASH_PATH = "phashes.pkl"

baseline_images_info = Path(os.path.expanduser("~/schoolwork/APS360 Deep Learning/fabscrape/cards.json"))
baseline_images_dir = Path(os.path.expanduser("~/schoolwork/APS360 Deep Learning/fabscrape/pngs"))

with open(baseline_images_info.resolve(), 'r') as file:
    cards = json.load(file)

def phash(img_path):
    return str(imagehash.phash(Image.open(img_path)))

def bytexor(a,b):
    return bytes(a ^ b for a, b in zip(a, b))

def distance(h1, h2):
    return bin(int.from_bytes(base64.b64decode(h1)) ^ int.from_bytes(base64.b64decode(h2))).count('1')

def np_to_img(img):
    return np.transpose(img, [1,2,0])
def img_to_np(img):
    return np.transpose(img, [2,0,1])
def np_to_pil(img):
    return Image.fromarray((np_to_img(img) * 255).astype('uint8'), 'RGB')

def get_image(img_path):
    img = Image.open(img_path).convert("RGB")

    img = np.array(img)
    img = np.transpose(img, [2,0,1])
    img = np.array(img/255, dtype=np.float32)

    return img

def show_points(img, points):
    plt.imshow(img)
    plt.scatter(points[:,0], points[:,1], c='red', s=40)

    for i, (x, y) in enumerate(points):
        plt.text(x, y, ' '+str(i), color='red')

    plt.show()

def transform_4point(img, points):
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

    #warped = warped[:maxy, :maxx]
    return warped


if os.path.exists(PHASH_PATH):
    with open(PHASH_PATH, 'rb') as file:
        phashes = pickle.load(file)
else:
    phashes = []
    for card in tqdm.tqdm(cards, total=len(cards)):
        filename = card["filename"].replace('.webp', '.png')
        filepath = (baseline_images_dir / filename).resolve()
        phashes.append((card["card_id"], phash(filepath)))
    with open(PHASH_PATH, 'wb') as file:
        pickle.dump(phashes, file)


def get_min(img):
    img = np_to_pil(img)

    imghash = str(imagehash.phash(img))

    #print('a', imghash)

    mindist = math.inf
    minimgs = []
    for cardid, cardhash in phashes:
        dist = distance(imghash, cardhash)
        if dist < mindist:
            mindist = dist
            minimgs = [cardid]
        elif dist == mindist:
            minimgs.append(cardid)

    return minimgs

def noise_coords(coords, noise_scale=4):
    xmax = 1024
    ymax = 1024
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


def ranking(img, top=5):
    img = np_to_pil(img)
    imghash = str(imagehash.phash(img))

    tosort = copy.deepcopy(phashes)
    tosort.sort(key=lambda x: distance(imghash, x[1]))

    return [i[0] for i in tosort[:top]]


images_info = Path(os.path.expanduser("~/schoolwork/APS360 Deep Learning/data_generation/cards.json"))
images_dir = Path(os.path.expanduser("~/schoolwork/APS360 Deep Learning/data_generation/images"))

#images_info = Path(os.path.expanduser("~/schoolwork/APS360 Deep Learning/real_data/cards.json"))
#images_dir = Path(os.path.expanduser("~/schoolwork/APS360 Deep Learning/real_data/cropped"))

with open(images_info.resolve(), 'r') as file:
    data = json.load(file)

results = []
rank_results = []

for i in tqdm.tqdm(data, total=len(data)):
    img_path = (images_dir / i['filename']).resolve()
    img = get_image(img_path)

    img = np_to_img(img)

    coords = i['coords']
    #print(coords)
    for j in range(4):
        coords[j][1] = 1024 - coords[j][1]
    corners = np.array(coords)

    #show_points(img, corners)

    #corners = noise_coords(corners)
    cropped = transform_4point(img, corners)
    #cropped = img

    #plt.imshow(img)
    #plt.show()
    #
    #plt.imshow(cropped)
    #plt.show()

    cropped = img_to_np(cropped)

    #print(get_min(img))

    #print(get_min(cropped))
    #print(i['card_id'])
    #print()

    mins = get_min(cropped)

    ranked = ranking(cropped)
    if i['card_id'] in mins:
        results.append(1/len(mins))
    else:
        results.append(0)

    if i['card_id'] in ranked:
        rank_results.append(1)
    else:
        #print(i['card_id'])
        #print(ranked)
        #print()
        rank_results.append(0)

print(sum(results)/len(results))
print(sum(rank_results)/len(rank_results))

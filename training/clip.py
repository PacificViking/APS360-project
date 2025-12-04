import os
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
from PIL import Image
import faiss

import json
import tqdm
import cv2
import matplotlib.pyplot as plt

from transformers import CLIPProcessor, CLIPModel
import pickle as pkl

from load import *

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embedding(img):
    #print(img.shape)
    img = np_to_pil(img)
    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        emb = model.get_image_features(**inputs)

    emb = emb/emb.norm(dim=-1, keepdim=True)
    toret = emb.cpu().numpy().astype("float32")[0]
    return toret


embeddings = []
if not os.path.exists("cardindex.pkl"):
    for i in tqdm.tqdm(card_data):
        img = get_image(os.path.join(card_images_path, i['filename']), fullsize=True)
        embeddings.append((i['card_id'], embedding(img)))
    with open('cardindex.pkl', 'wb') as file:
        pkl.dump(embeddings, file)
else:
    with open('cardindex.pkl', 'rb') as file:
        embeddings = pkl.load(file)

def query(img):
    emb = embedding(img)
    #print(emb)
    sortt = sorted(embeddings, key=lambda x: -np.dot(x[1], emb))
    return [i[0] for i in sortt]


BENCH = 0

if BENCH:
    count0 = 0
    count1 = 0
    count5 = 0

    for i in tqdm.tqdm(image_data, total=len(image_data)):
        img_path = os.path.join(image_path, i['filename'])
        img = get_image(img_path, fullsize=True)
        coords = get_coords_from(i, fullsize=True)
        coords = noise_coords(coords, fullsize=True)
        img = np_to_img(img)
        img = transform_4point(img, coords)
        img = img_to_np(img)

        #np_to_pil(img).show()
        #input()
        ans = query(img)
        count0 += 1
        if i['card_id'] == ans[0]:
            count1 += 1
        if i['card_id'] in ans[:5]:
            count5 += 1

        #print(i['card_id'], ans[:5])

    print(1-(count1/count0))
    print(1-(count5/count0))

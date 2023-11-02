# 加载模型
import gzip
import json
import os
from urllib.parse import urlparse
import re
import cn_clip
import cn_clip.clip as clip
import pandas as pd
from cn_clip.clip import load_from_name, available_models, load
import torch
from PIL import Image
import numpy as np

print("Available models:", available_models())
# 加载预训练的CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cuda:0'
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./cn_CLIP_model')  #
# model = model.to(device)
# print(model.device)
device1 = next(model.parameters()).device
print(device1)

model.eval()


data_name = "meta_All_Beauty.json.gz"


def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)


item_df = pd.DataFrame(parse(data_name))

text_features_list = []
image_features_list = []

# # 给定的URL
# url = "https://images-na.ssl-images-amazon.com/images/I/31wxN2ktk0L._SL500_AC_SS350_.jpg"
# parts = url.split("/")
# filename1 = parts[-1]
# # 如果您希望去掉文件扩展名（例如 ".jpg"），可以使用 splittext 方法
# filename_without_extension, extension = os.path.splitext(filename[0])
# print(filename1)
# print(filename_without_extension)
# local_path = 'images_HighRes\\' + filename1
# print(local_path)


def extract_filename_from_url(url):
    return os.path.basename(urlparse(url).path)


for index, item in item_df.iterrows():
    # print(index,item)
    # 处理文本和图片
    text = clip.tokenize(item.title)
    filename = item.imageURLHighRes

    if not filename:
        continue
    parts = filename[0].split("/")
    fname = extract_filename_from_url(parts[-1])
    try:
        image = preprocess(Image.open('images_HighRes\\' + fname)).unsqueeze(0).to(device)
    except IOError:
        print(f"Cannot open image from URL: {fname}")
        continue
    text = text.to('cuda:0')
    image = image.to ('cuda:0')
    # print(fname)
    # image = preprocess(Image.open('images_HighRes' + filename)).unsqueeze(0).to(device)

    with torch.no_grad():
        # embedding图片和文本

        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_features_list.append(text_features)
        image_features_list.append(image_features)

torch.save(image_features_list, 'image_embedding.pt')
torch.save(text_features_list, 'text_embedding.pt')

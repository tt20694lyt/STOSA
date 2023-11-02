import gzip
import json
import os
import requests
from PIL import Image
from io import BytesIO


# 确保图片保存的目录存在
if not os.path.exists('images'):
    os.makedirs('images')

data_name = "meta_All_Beauty.json.gz"


def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)


image_urls = []  # 初始化空列表

for one_interaction in parse(data_name):
    img = one_interaction['imageURLHighRes']
    # print(img)
    image_urls.append(img)  # 将img添加到列表中
# print(image_urls)


def download_and_save_image(url, save_dir="images_HighRes"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        response = requests.get(url)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))
        save_path = os.path.join(save_dir, url.split("/")[-1])
        img.save(save_path)
        print(f"Image saved to {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {str(e)}")


for img_list in image_urls:
    if isinstance(img_list, list) and img_list:
        download_and_save_image(img_list[0])

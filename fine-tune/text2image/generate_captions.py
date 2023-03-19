"""
使用 BLIP-2 模型生成 image caption
paper: [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)
HuggingFace: [Salesforce/blip2-opt-6.7b](https://huggingface.co/Salesforce/blip2-opt-6.7b)
调用 BLIP-2 要求 transformers 库的版本 >= 4.27

该注释写于2023.03.18:
    目前可用 pip install transformers 更新的版本均为稳定版, 最高为4.26.1, 但开发版目前已到 4.27.0 版
    若当前 transformers 最高版本还未到 4.27, 使用 pip install git+https://github.com/huggingface/transformers 命令直接安装开发版
"""

import os
import sys
import json
from tqdm import tqdm
from PIL import Image
from transformers.utils import check_min_version
from transformers import Blip2Processor, Blip2ForConditionalGeneration


check_min_version("4.27.0.dev0")  # transformers库版本>=4.27, 否则无法使用 BLIP-2, 具体请查看最上方注释

# image-to-text model
# BLIP-2 模型加载比较慢, 需要等待较长时间, 且显存需要30G
# 若显存不足可使用 Salesforce/blip2-opt-2.7b, 或者其他模型(如BLIP等), 但效果可能会稍差一些
print("Loading model...")
device = "cuda:1"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b").to(device)
print("Model loading completed.")

image_folder = os.path.join(sys.path[0], 'images')

image_caption_dic = {}

for image_name in tqdm(os.listdir(image_folder)):
    image_path = os.path.join(image_folder, image_name)
    image = Image.open(image_path)
    # generate image caption
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, 
                         max_new_tokens=64, 
                         num_beams=5  # 使用 beam search 要比 greedy search 的效果要好
                        )             # 其他参数根据自己需要添加/修改
    image_caption = processor.decode(out[0], skip_special_tokens=True).replace('\n', '')
    image_caption_dic[image_path] = image_caption
    
# 写入json文件
json_path = os.path.join(sys.path[0], 'image_captions.json')  # 输出文件的路径
with open(json_path, 'w', encoding='utf-8') as fout:
    json.dump(image_caption_dic, fout, indent=4)

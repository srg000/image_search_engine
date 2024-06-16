from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
import faiss
import json
from tqdm import tqdm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("mps")

model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

d = 512  # 向量维度为512
index = faiss.IndexFlatL2(d)  # 使用 L2 距离
# 文件夹路径
folder_path = './val_images'


def init_data() -> dict:
    # 遍历文件夹
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否为图片文件（这里简单地检查文件扩展名）
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    id2filename = {idx: x for idx, x in enumerate(file_paths)}

    # 保存为 JSON 文件
    with open('id2filename.json', 'w') as json_file:
        json.dump(id2filename, json_file)

    for file_path in tqdm(file_paths, total=len(file_paths)):
        # 使用PIL打开图片
        image = Image.open(file_path)
        inputs = processor(images=image, return_tensors="pt", padding=True)
        image_features = model.get_image_features(inputs["pixel_values"])
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize
        image_features = image_features.detach().numpy()
        index.add(image_features)
        # 关闭图像，释放资源
        image.close()
    print(f"索引库数量：{index.ntotal}")
    return id2filename


def text_search(text, k=1):
    inputs = processor(text=text, images=None, return_tensors="pt", padding=True)
    text_features = model.get_text_features(inputs["input_ids"], inputs["attention_mask"])
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # normalize
    text_features = text_features.detach().numpy()
    D, I = index.search(text_features, k)  # 实际的查询
    filenames = [[id2filename[j] for j in i] for i in I]

    return text, D, filenames


def image_search(img_path, id2filename, k=1):
    image = Image.open(img_path)
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize

    image_features = image_features.detach().numpy()
    D, I = index.search(image_features, k)  # 实际的查询

    filenames = [[id2filename[j] for j in i] for i in I]

    return img_path, D, filenames


def data_test():
    index = faiss.IndexFlatL2(512)  # 使用 L2 距离
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否为图片文件（这里简单地检查文件扩展名）
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    for file_path in tqdm(file_paths, total=len(file_paths)):
        # 使用PIL打开图片
        image = Image.open(file_path)
        inputs = processor(images=image, return_tensors="pt", padding=True)
        image_features = model.get_image_features(inputs["pixel_values"])
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize
        image_features = image_features.detach().numpy()
        index.add(image_features)
        # 关闭图像，释放资源
        image.close()
    print(f"索引库数量：{index.ntotal}")


if __name__ == "__main__":
    id2filename = init_data()
    # text = ["香蕉", "葡萄"]
    # text, D, filenames = text_search(text)
    # print(text, D, filenames)

    img_path = "./val_images/Image_4.jpg"
    img_path, D, filenames = image_search(img_path, id2filename, k=2)
    print(img_path, D, filenames)

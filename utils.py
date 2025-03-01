from tqdm import tqdm
import os
from PIL import Image

# 将AnimeFaceDataset内的图片全部转为png格式，再移动到AnimeFace64中，并且以顺序命名
def handle2dataset():
    # 定义文件夹路径
    anime_face_dataset_dir = "AnimeFaces64"  # 原始图片文件夹
    anime_faces64_dir = "new"  # 目标图片文件夹

    no=0
    # 遍历AnimeFaceDataset文件夹中的所有jpg文件
    for filename in tqdm(os.listdir(anime_face_dataset_dir), desc="Denoising", ncols=100):
        # 构建原始文件路径
        jpg_path = os.path.join(anime_face_dataset_dir, filename)
        with Image.open(jpg_path) as img:
            # 图片的宽度和高度小于64的直接跳过
            width, height = img.size
            if width < 64 or height < 64:
                continue

            # 生成新的文件名
            no += 1
            new_filename = f"{no}.png"
            new_path = os.path.join(anime_faces64_dir, new_filename)
            # 转换为png格式并保存
            img.save(new_path, "PNG")

    print("图片转换和重命名完成！")


import shutil
# 建立新文件夹，若已有则清空其内容
def newDir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)


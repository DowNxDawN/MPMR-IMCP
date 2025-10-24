import os
import time
import glob
import json
import shutil
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict
import random
from multiprocessing import Pool, cpu_count
import logging

# 设置输入文件夹和分割的像素大小
input_dir = "/path/to/your/data/wsi"
patch_size = 224  # 设置分割的像素大小
train_val_split = 0.8  # 设置训练集和验证集的划分比例
overlap_ratio = 0.5  # 设置分割图像时允许的重叠率
Image.MAX_IMAGE_PIXELS = None  # 规定大图像处理时的最大像素限制

"""
准备工作：
1、Labelme标注工具对JPEG格式图像进行标注，生成JSON文件。
2、将图像和注释文件放在同一个路径下，并在上方填写正确的路径。
3、设置分割的像素大小以及训练集/验证集的比例、分割重叠率、像素限制。
    输入文件夹格式架构如下：
        INPUT_DIR/
        ├── image1.jpg
        ├── image1.json
        ├── image2.jpg
        ├── image2.json
        └── ...
运行结果：
0、输出所有Json文件的标签统计信息和尺寸信息，并询问是否进行标签转换。
    输出信息包括：
        1)共有多少类别的标签
        2)共有多少个标签
        3)每类标签的数量
        4)最大和最小的像素尺寸
        5)平均像素尺寸
1、得到一个名为"patches"的输出文件夹，包含了所有裁剪后的patches。
2、patches按照标签进行分类，每个标签一个文件夹，文件夹下包含train和val两个子文件夹。
3、每个patch的命名格式为"image_index_shape_index_patch_index_label.png"。
    输出文件夹格式架构如下：
        OUTPUT_DIR/
        ├── label1/
        │   ├── sublabel1/
        │   │   ├── train/
        │   │   │   ├── image1_1_1_label1.png
        │   │   │   ├── image1_1_2_label1.png
        │   │   │   └── ...
        │   │   └── val/
        │   │       ├── image1_1_1_label1.png
        │   │       ├── image1_1_2_label1.png
        │   │       └── ...
        │   ├── sublabel2/
        │   │   ├── train/
        │   │   │   ├── image1_2_1_label1.png
        │   │   │   ├── image1_2_2_label1.png
        │   │   │   └── ...
        │   │   └── val/
        │   │       ├── image1_2_1_label1.png
        │   │       ├── image1_2_2_label1.png
        │   │       └── ...
        ├── label2/
        │   └── ...
        └── ...
"""

# 配置日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_image_and_masks(json_file, input_dir, target_label):
    with open(json_file, "r") as f:
        data = json.load(f)
    image_filename = os.path.basename(data["imagePath"].replace("\\", "/"))
    image_path_jpg = os.path.join(input_dir, image_filename)
    image_path_jpeg = os.path.join(input_dir, image_filename.replace(".jpg", ".jpeg"))

    if not os.path.exists(image_path_jpg) and not os.path.exists(image_path_jpeg):
        logging.error(f"文件不存在：{image_path_jpg} 或 {image_path_jpeg}")
        return None, None

    image_path = image_path_jpg if os.path.exists(image_path_jpg) else image_path_jpeg
    logging.info(f"正在加载图像：{image_path}")

    image = Image.open(image_path)
    masks = defaultdict(lambda: Image.new("L", image.size, 0))
    other_mask = Image.new("L", image.size, 1)  # 初始化 "Other" 类掩码，覆盖整个图像
    other_has_label = False  # 标记 "Other" 类别是否有标签

    for shape in data["shapes"]:
        label = shape["label"]
        points = np.array(shape["points"], dtype=np.int32)
        if label == target_label:
            ImageDraw.Draw(masks[label]).polygon(points.flatten().tolist(), outline=1, fill=1)
            ImageDraw.Draw(other_mask).polygon(
                points.flatten().tolist(), outline=0, fill=0
            )  # 从 "Other" 类掩码中移除标注区域
        else:
            ImageDraw.Draw(masks["Other"]).polygon(
                points.flatten().tolist(), outline=1, fill=1
            )  # 将其他类别区域标注为 "Other"
            other_has_label = True  # 标记 "Other" 类别有标签

    masks["Other"] = other_mask  # 添加 "Other" 类掩码到掩码字典中
    return image, {label: (np.array(mask), other_has_label) for label, mask in masks.items()}


def split_image(image, masks, patch_size, overlap=0.5):
    patches = defaultdict(list)
    step = int(patch_size * (1 - overlap))
    for y in range(0, image.shape[0] - patch_size + 1, step):
        for x in range(0, image.shape[1] - patch_size + 1, step):
            patch = image[y : y + patch_size, x : x + patch_size]
            for label, (mask, other_has_label) in masks.items():
                patch_mask = mask[y : y + patch_size, x : x + patch_size]
                if np.sum(patch_mask) > 0.5 * patch_size * patch_size:
                    patches[label].append((patch, other_has_label))
    return patches


def save_patches(patches, output_dir, json_file, split_type):
    for label, (label_patches, other_has_label) in patches.items():
        split_dir = os.path.join(output_dir, split_type, label)

        os.makedirs(split_dir, exist_ok=True)

        if label == "Other":
            if other_has_label:
                # 随机抽样，选择十分之一的 "Other" 类别 patches
                label_patches = random.sample(label_patches, len(label_patches) // 10)
            else:
                # 随机抽样，选择百分之一的 "Other" 类别 patches
                label_patches = random.sample(label_patches, len(label_patches) // 100)

        for k, patch in enumerate(label_patches, 1):
            patch_image = Image.fromarray(patch)
            patch_image.save(os.path.join(split_dir, f"{os.path.basename(json_file)[:-5]}_{k}.png"))


def process_json_file(json_file, split_type, target_label):
    logging.info(f"正在处理：{os.path.basename(json_file)}")
    image, masks = load_image_and_masks(json_file, input_dir, target_label)
    if image is None or masks is None:
        return 0
    image = np.array(image)
    patches = split_image(image, masks, patch_size)
    save_patches(patches, output_dir, json_file, split_type)
    logging.info(f"{os.path.basename(json_file)}处理完毕")
    return (sum(len(p) for p in patches.values()),)


def process_files_sequentially(json_files, split_type, target_label):
    num_patches = 0
    for json_file in json_files:
        num_patches += process_json_file(json_file, split_type, target_label)[0]
    return num_patches


def process_files_in_parallel(json_files, num_processes, split_type, target_label):
    with Pool(processes=num_processes) as pool:
        num_patches = sum(
            result[0]
            for result in pool.starmap(
                process_json_file, [(json_file, split_type, target_label) for json_file in json_files]
            )
        )
    return num_patches


def get_label_files(json_files, target_label):
    label_files = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)
            for shape in data.get("shapes", []):
                label = shape.get("label")
                if label == target_label:
                    label_files.append(json_file)
                    break  # 假设每个文件只有一个主要标签
    return label_files


def split_files(label_files, train_val_split):
    random.shuffle(label_files)
    split_index = int(len(label_files) * train_val_split)
    train_files = label_files[:split_index]
    val_files = label_files[split_index:]
    return train_files, val_files


def main():
    start_time = time.time()  # 记录开始时间
    directory = input_dir
    target_label = input("请输入要分割的目标标签: ").strip()

    # 根据目标标签设置输出文件夹
    global output_dir
    output_dir = os.path.join(os.path.dirname(input_dir), f"patches_{target_label}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    label_files = get_label_files(json_files, target_label)
    train_files, val_files = split_files(label_files, train_val_split)

    use_multiprocessing = input("是否使用多进程处理？(y/n): ").strip().lower() == "y"

    if use_multiprocessing:
        num_processes = int(input("请输入要使用的进程数量: ").strip())
        num_train_patches = process_files_in_parallel(train_files, num_processes, "train", target_label)
        num_val_patches = process_files_in_parallel(val_files, num_processes, "val", target_label)
    else:
        num_train_patches = process_files_sequentially(train_files, "train", target_label)
        num_val_patches = process_files_sequentially(val_files, "val", target_label)

    logging.info(
        f"处理完毕，共完成{input_dir}文件夹下{len(json_files)}张图像处理，共生成{num_train_patches}张训练patches和{num_val_patches}张验证patches"
    )

    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算总用时
    logging.info(f"总用时: {total_time:.2f} 秒")  # 输出总用时


if __name__ == "__main__":
    main()

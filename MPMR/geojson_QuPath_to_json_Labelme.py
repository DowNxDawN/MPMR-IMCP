import json
import os
from PIL import Image
from tqdm import tqdm

# 允许处理大图像
Image.MAX_IMAGE_PIXELS = None


def convert_geojson_to_labelme(geojson_file):
    # 读取 GeoJSON 文件
    with open(geojson_file, "r") as f:
        geojson_data = json.load(f)

    # 初始化 LabelMe 格式的 JSON 数据结构
    labelme_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": "",
        "imageData": None,
        "imageHeight": 0,
        "imageWidth": 0,
    }

    # 提取 GeoJSON 文件中的注释信息并转换为 LabelMe 格式
    for feature in geojson_data["features"]:
        if feature["geometry"]["type"] == "Polygon":
            points = feature["geometry"]["coordinates"][0]
            shape = {
                "label": feature["properties"]["classification"]["name"],
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
            labelme_data["shapes"].append(shape)

    # 假设图像文件名与 GeoJSON 文件名相同，但扩展名可能不同
    base_filename = os.path.basename(geojson_file).replace(".geojson", "")
    possible_extensions = [".png", ".jpg", ".jpeg", ".svs"]  # 添加 .svs 扩展名
    image_filename = None

    for ext in possible_extensions:
        potential_image_path = os.path.join(os.path.dirname(geojson_file), base_filename + ext)
        if os.path.exists(potential_image_path):
            image_filename = base_filename + ext
            break

    if image_filename is None:
        print(f"未找到与 {geojson_file} 对应的图像文件")
        return

    labelme_data["imagePath"] = image_filename

    # 自动识别图像的尺寸
    image_path = os.path.join(os.path.dirname(geojson_file), image_filename)
    with Image.open(image_path) as img:
        labelme_data["imageHeight"] = img.height
        labelme_data["imageWidth"] = img.width

    # 保存转换后的 JSON 文件到与原始 GeoJSON 文件相同的目录
    output_file = geojson_file.replace(".geojson", ".json")
    with open(output_file, "w") as f:
        json.dump(labelme_data, f, indent=4)

    # 打印转换成功的信息
    print(f"转换成功: {output_file}")


def batch_convert_geojson_to_labelme(input_dir):
    # 获取输入目录中的所有 GeoJSON 文件
    geojson_files = [f for f in os.listdir(input_dir) if f.endswith(".geojson")]

    # 使用 tqdm 显示进度条
    for filename in tqdm(geojson_files, desc="转换进度"):
        geojson_file = os.path.join(input_dir, filename)
        convert_geojson_to_labelme(geojson_file)


# 示例用法
input_dir = "/path/to/your/data"
print(f"开始转换目录: {input_dir}")
batch_convert_geojson_to_labelme(input_dir)
print("所有文件转换完成")

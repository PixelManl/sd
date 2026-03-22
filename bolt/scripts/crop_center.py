import os
import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

# ==================== 配置参数 ====================
IMG_DIR = "D:\桌面文件\编程\比赛\挂点金具螺栓缺失\images"          # 原始图像文件夹
XML_DIR = "D:\桌面文件\编程\比赛\挂点金具螺栓缺失-xml\挂点金具螺栓缺失-xml"            # XML标注文件夹（如果与图像同文件夹可设相同）
OUT_DIR = "D:\桌面文件\编程\比赛\挂点金具螺栓缺失\\512images"         # 输出裁剪图像文件夹
CROP_SIZE = 512                     # 裁剪尺寸（正方形）
FILL_COLOR = (0, 0, 0)              # 填充颜色（黑色）

# 可选：是否将裁剪区域可视化并保存（用于调试）
VISUALIZE_BBOX = False              # 是否在裁剪图像上绘制原始边界框（调试用）

# ==================== 辅助函数 ====================
def parse_voc_xml(xml_path):
    """解析 VOC XML 文件，返回对象列表，每个对象为 (class_name, (xmin, ymin, xmax, ymax))"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append((name, (xmin, ymin, xmax, ymax)))
    return objects

def crop_center_with_padding(image, center_x, center_y, crop_size, fill_color):
    """
    以指定中心裁剪正方形区域，若超出边界则用填充色填充。
    返回裁剪后的 PIL Image。
    """
    img_w, img_h = image.size
    half = crop_size // 2

    # 计算裁剪区域的边界（可能为负或超出）
    left = center_x - half
    top = center_y - half
    right = center_x + half
    bottom = center_y + half

    # 创建新画布（填充色）
    cropped_img = Image.new(image.mode, (crop_size, crop_size), fill_color)

    # 计算原图区域与裁剪区域的重叠部分
    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(img_w, right)
    src_bottom = min(img_h, bottom)

    # 计算目标画布上的对应位置
    dst_left = src_left - left
    dst_top = src_top - top
    dst_right = src_right - left
    dst_bottom = src_bottom - top

    if src_right > src_left and src_bottom > src_top:
        # 裁剪原图有效区域
        region = image.crop((src_left, src_top, src_right, src_bottom))
        cropped_img.paste(region, (dst_left, dst_top, dst_right, dst_bottom))

    return cropped_img

def visualize_bbox_on_crop(cropped_img, original_bbox, center, crop_size):
    """
    在裁剪图像上绘制原始边界框（用于调试）。
    original_bbox: (xmin, ymin, xmax, ymax) 原图坐标
    center: (cx, cy) 原图坐标
    crop_size: 裁剪尺寸
    """
    half = crop_size // 2
    # 计算裁剪区域在原图中的偏移量
    left = center[0] - half
    top = center[1] - half
    # 将原始边界框映射到裁剪图坐标
    xmin = original_bbox[0] - left
    ymin = original_bbox[1] - top
    xmax = original_bbox[2] - left
    ymax = original_bbox[3] - top
    # 绘制矩形
    draw = ImageDraw.Draw(cropped_img)
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
    return cropped_img

# ==================== 主处理流程 ====================
def main():
    # 创建输出文件夹
    os.makedirs(OUT_DIR, exist_ok=True)

    # 获取所有图像文件（假设图像格式为 .jpg/.png，可根据需要扩展）
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(image_extensions)]
    total_crops = 0

    for img_file in image_files:
        # 构建对应的 XML 文件路径（假设 XML 与图像同名）
        base_name = os.path.splitext(img_file)[0]
        xml_path = os.path.join(XML_DIR, base_name + ".xml")
        if not os.path.exists(xml_path):
            print(f"警告: 找不到 {xml_path}，跳过 {img_file}")
            continue

        # 读取图像
        img_path = os.path.join(IMG_DIR, img_file)
        try:
            image = Image.open(img_path).convert("RGB")  # 统一为RGB
        except Exception as e:
            print(f"错误: 无法读取图像 {img_path}: {e}")
            continue

        # 解析 XML
        objects = parse_voc_xml(xml_path)
        if not objects:
            print(f"警告: {xml_path} 中没有对象，跳过")
            continue

        # 对每个边界框生成裁剪图像
        for idx, (class_name, bbox) in enumerate(objects):
            xmin, ymin, xmax, ymax = bbox
            # 计算边界框中心
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            # 裁剪
            cropped = crop_center_with_padding(image, cx, cy, CROP_SIZE, FILL_COLOR)

            # 可选：在裁剪图上绘制原始边界框（调试）
            if VISUALIZE_BBOX:
                cropped = visualize_bbox_on_crop(cropped, bbox, (cx, cy), CROP_SIZE)

            # 保存文件
            out_filename = f"{base_name}_{idx+1}_{class_name}.png"
            out_path = os.path.join(OUT_DIR, out_filename)
            cropped.save(out_path)
            print(f"已保存: {out_path}")
            total_crops += 1

    print(f"\n处理完成！共生成 {total_crops} 个裁剪图像，保存在 {OUT_DIR}")

if __name__ == "__main__":
    main()
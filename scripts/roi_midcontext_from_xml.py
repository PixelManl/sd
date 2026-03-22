import os
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2

"""
缺陷中近景ROI
"""

IMAGE_DIR = 'defect_images'
XML_DIR = 'defect_xml'
OUT_DIR = 'train_data_mid/6_missing_fastener'
MID_SCALE = 5.5
OUT_SIZE = 512
IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
CAPTION = 'missing nut defect, power grid equipment, inspection scene, metal structure'


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def find_image_path(stem: str):
    for ext in IMG_EXTS:
        p = Path(IMAGE_DIR) / f'{stem}{ext}'
        if p.exists():
            return str(p)
    return None


def parse_voc(xml_path):
    root = ET.parse(xml_path).getroot()
    filename = root.findtext('filename')
    objs = []
    for obj in root.findall('object'):
        b = obj.find('bndbox')
        xmin = int(float(b.findtext('xmin')))
        ymin = int(float(b.findtext('ymin')))
        xmax = int(float(b.findtext('xmax')))
        ymax = int(float(b.findtext('ymax')))
        name = obj.findtext('name', default='missing_fastener')
        objs.append((name, xmin, ymin, xmax, ymax))
    return filename, objs


def square_crop(img, box, scale=5.5, out_size=512):
    h, w = img.shape[:2]
    _, x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(bw, bh) * scale
    sx1 = int(round(cx - side / 2))
    sy1 = int(round(cy - side / 2))
    sx2 = int(round(cx + side / 2))
    sy2 = int(round(cy + side / 2))

    pad_left = max(0, -sx1)
    pad_top = max(0, -sy1)
    pad_right = max(0, sx2 - w)
    pad_bottom = max(0, sy2 - h)

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        img = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101
        )
        sx1 += pad_left
        sx2 += pad_left
        sy1 += pad_top
        sy2 += pad_top

    crop = img[sy1:sy2, sx1:sx2]
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop


def main():
    ensure_dir(OUT_DIR)
    xml_files = sorted(Path(XML_DIR).glob('*.xml'))
    count = 0
    for xml_path in xml_files:
        stem = xml_path.stem
        filename, objs = parse_voc(str(xml_path))
        image_path = None
        if filename:
            p = Path(IMAGE_DIR) / filename
            if p.exists():
                image_path = str(p)
        if image_path is None:
            image_path = find_image_path(stem)
        if image_path is None:
            print(f'[WARN] image not found for {xml_path.name}')
            continue
        img = cv2.imread(image_path)
        if img is None:
            print(f'[WARN] failed to read {image_path}')
            continue

        for i, box in enumerate(objs):
            crop = square_crop(img, box, scale=MID_SCALE, out_size=OUT_SIZE)
            base = f'{stem}_{i:02d}_mid'
            cv2.imwrite(str(Path(OUT_DIR) / f'{base}.png'), crop)
            with open(Path(OUT_DIR) / f'{base}.txt', 'w', encoding='utf-8') as f:
                f.write(CAPTION)
            count += 1
    print(f'[DONE] mid-context ROI samples: {count}')


if __name__ == '__main__':
    main()

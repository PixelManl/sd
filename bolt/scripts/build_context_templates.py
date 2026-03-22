import os
import json
import cv2
import math
import xml.etree.ElementTree as ET
from pathlib import Path

"""
由缺陷 XML 自动生成上下文模板
每个缺陷框会生成：
    templates/images/xxx.png 上下文模板图
    templates/masks/xxx.png 挖空 mask，中间缺陷区是 0，外围上下文是 255
    templates/meta/xxx.json
        记录：
            原 bbox
            ROI 大小
            bbox 在 ROI 里的中心
            后续如何把匹配位置映射回“健康螺帽候选框”

"""

# ========= 可改参数 =========
DEFECT_IMG_DIR = "defect_images"
DEFECT_XML_DIR = "defect_xml"

OUT_TEMPLATE_IMG_DIR = "templates/images"
OUT_TEMPLATE_MASK_DIR = "templates/masks"
OUT_TEMPLATE_META_DIR = "templates/meta"

# 以缺陷框为中心，向外扩多少倍
CONTEXT_SCALE = 3.0

# 模板最小尺寸，避免特别小的框生成没意义模板
MIN_TEMPLATE_SIZE = 48

# 中心缺陷区域再额外放大多少，用于“挖空”
CENTER_HOLE_SCALE = 1.15

# 是否统一保存为灰度模板
SAVE_GRAY = True
# ==========================


def ensure_dirs():
    for d in [OUT_TEMPLATE_IMG_DIR, OUT_TEMPLATE_MASK_DIR, OUT_TEMPLATE_META_DIR]:
        os.makedirs(d, exist_ok=True)


def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    size_node = root.find("size")
    width = int(size_node.findtext("width"))
    height = int(size_node.findtext("height"))

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.findtext("xmin")))
        ymin = int(float(bnd.findtext("ymin")))
        xmax = int(float(bnd.findtext("xmax")))
        ymax = int(float(bnd.findtext("ymax")))
        objects.append({
            "name": name,
            "bbox": [xmin, ymin, xmax, ymax]
        })

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects
    }


def clamp(v, low, high):
    return max(low, min(v, high))


def build_context_roi(img, bbox, context_scale=3.0, center_hole_scale=1.15):
    h, w = img.shape[:2]
    xmin, ymin, xmax, ymax = bbox

    bw = xmax - xmin
    bh = ymax - ymin
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    # 上下文 ROI 尺寸
    roi_w = max(int(bw * context_scale), MIN_TEMPLATE_SIZE)
    roi_h = max(int(bh * context_scale), MIN_TEMPLATE_SIZE)

    rx1 = int(round(cx - roi_w / 2))
    ry1 = int(round(cy - roi_h / 2))
    rx2 = int(round(cx + roi_w / 2))
    ry2 = int(round(cy + roi_h / 2))

    rx1 = clamp(rx1, 0, w - 1)
    ry1 = clamp(ry1, 0, h - 1)
    rx2 = clamp(rx2, 1, w)
    ry2 = clamp(ry2, 1, h)

    roi = img[ry1:ry2, rx1:rx2].copy()

    # 缺陷框映射到 ROI 局部坐标
    local_xmin = xmin - rx1
    local_ymin = ymin - ry1
    local_xmax = xmax - rx1
    local_ymax = ymax - ry1

    # 中心“挖空”区域略放大
    lc_cx = (local_xmin + local_xmax) / 2.0
    lc_cy = (local_ymin + local_ymax) / 2.0
    lc_w = (local_xmax - local_xmin) * center_hole_scale
    lc_h = (local_ymax - local_ymin) * center_hole_scale

    hx1 = int(round(lc_cx - lc_w / 2))
    hy1 = int(round(lc_cy - lc_h / 2))
    hx2 = int(round(lc_cx + lc_w / 2))
    hy2 = int(round(lc_cy + lc_h / 2))

    rh, rw = roi.shape[:2]
    hx1 = clamp(hx1, 0, rw - 1)
    hy1 = clamp(hy1, 0, rh - 1)
    hx2 = clamp(hx2, 1, rw)
    hy2 = clamp(hy2, 1, rh)

    # mask: 外围上下文 255，中间挖空 0
    mask = 255 * (roi[:, :, 0] if roi.ndim == 3 else roi).astype("uint8")
    mask[:] = 255
    mask[hy1:hy2, hx1:hx2] = 0

    # 记录模板中心与原 bbox 的关系
    meta = {
        "roi_global_box": [rx1, ry1, rx2, ry2],
        "bbox_global": [xmin, ymin, xmax, ymax],
        "bbox_local": [local_xmin, local_ymin, local_xmax, local_ymax],
        "hole_local": [hx1, hy1, hx2, hy2],
        "roi_size": [rw, rh],
        "bbox_size": [bw, bh],
        "bbox_center_in_roi": [
            (local_xmin + local_xmax) / 2.0,
            (local_ymin + local_ymax) / 2.0
        ]
    }

    return roi, mask, meta


def main():
    ensure_dirs()

    xml_files = sorted(Path(DEFECT_XML_DIR).glob("*.xml"))
    template_count = 0

    for xml_path in xml_files:
        ann = parse_voc_xml(str(xml_path))
        img_path = os.path.join(DEFECT_IMG_DIR, ann["filename"])
        if not os.path.exists(img_path):
            print(f"[WARN] image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] failed to read image: {img_path}")
            continue

        if SAVE_GRAY:
            img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_proc = img

        stem = Path(ann["filename"]).stem

        for idx, obj in enumerate(ann["objects"]):
            bbox = obj["bbox"]
            name = obj["name"]

            xmin, ymin, xmax, ymax = bbox
            if xmax <= xmin or ymax <= ymin:
                continue

            roi, mask, meta = build_context_roi(
                img_proc, bbox,
                context_scale=CONTEXT_SCALE,
                center_hole_scale=CENTER_HOLE_SCALE
            )

            rh, rw = roi.shape[:2]
            if min(rw, rh) < MIN_TEMPLATE_SIZE:
                continue

            out_name = f"{stem}_{idx:02d}"

            img_out = os.path.join(OUT_TEMPLATE_IMG_DIR, out_name + ".png")
            mask_out = os.path.join(OUT_TEMPLATE_MASK_DIR, out_name + ".png")
            meta_out = os.path.join(OUT_TEMPLATE_META_DIR, out_name + ".json")

            cv2.imwrite(img_out, roi)
            cv2.imwrite(mask_out, mask)

            meta["source_image"] = ann["filename"]
            meta["source_xml"] = xml_path.name
            meta["object_name"] = name
            meta["template_name"] = out_name

            with open(meta_out, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            template_count += 1

    print(f"[DONE] generated {template_count} context templates.")


if __name__ == "__main__":
    main()
import os
import json
import cv2
import math
import numpy as np
from pathlib import Path

"""
在健康图中做挖空模板匹配
对每张健康图输出：
    match_results/json/xxx.json 候选健康螺帽位置列表
    match_results/vis/xxx.jpg 可视化框图
"""

# ========= 可改参数 =========
HEALTHY_IMG_DIR = "healthy_images"

TEMPLATE_IMG_DIR = "templates/images"
TEMPLATE_MASK_DIR = "templates/masks"
TEMPLATE_META_DIR = "templates/meta"

OUT_VIS_DIR = "match_results/vis"
OUT_JSON_DIR = "match_results/json"

# 多尺度模板匹配的缩放范围
SCALES = [0.8, 0.9, 1.0, 1.1, 1.2]

# 带 mask 的匹配方法：官方支持 TM_SQDIFF 和 TM_CCORR_NORMED
MATCH_METHOD = cv2.TM_CCORR_NORMED

# 分数阈值
SCORE_THRESH = 0.55

# 每张图保留前多少个结果
TOPK = 20

# NMS 阈值
NMS_IOU_THRESH = 0.3

# 边缘安全距离比例
EDGE_MARGIN_RATIO = 0.03

# 是否在匹配前做 CLAHE
USE_CLAHE = True
# ==========================


def ensure_dirs():
    os.makedirs(OUT_VIS_DIR, exist_ok=True)
    os.makedirs(OUT_JSON_DIR, exist_ok=True)


def preprocess_gray(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def compute_sharpness(gray, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(cv2.Laplacian(patch, cv2.CV_64F).var())


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6

    return inter / union


def nms(cands, iou_thresh=0.3):
    cands = sorted(cands, key=lambda x: x["score"], reverse=True)
    keep = []
    for c in cands:
        ok = True
        for k in keep:
            if bbox_iou(c["pred_box"], k["pred_box"]) > iou_thresh:
                ok = False
                break
        if ok:
            keep.append(c)
    return keep


def load_templates():
    templates = []
    img_paths = sorted(Path(TEMPLATE_IMG_DIR).glob("*.png"))

    for img_path in img_paths:
        name = img_path.stem
        mask_path = Path(TEMPLATE_MASK_DIR) / f"{name}.png"
        meta_path = Path(TEMPLATE_META_DIR) / f"{name}.json"

        if not mask_path.exists() or not meta_path.exists():
            continue

        templ = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        templates.append({
            "name": name,
            "templ": templ,
            "mask": mask,
            "meta": meta
        })

    return templates


def resize_template_pack(templ, mask, meta, scale):
    h, w = templ.shape[:2]
    nw = max(8, int(round(w * scale)))
    nh = max(8, int(round(h * scale)))

    templ_r = cv2.resize(templ, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    # bbox_center_in_roi 跟着缩放
    cx, cy = meta["bbox_center_in_roi"]
    bw, bh = meta["bbox_size"]

    cx *= scale
    cy *= scale
    bw *= scale
    bh *= scale

    meta_r = {
        "bbox_center_in_roi": [cx, cy],
        "bbox_size": [bw, bh]
    }
    return templ_r, mask_r, meta_r


def template_match_with_mask(gray, templ, mask):
    # 官方：带 mask 只建议少数方法，这里用 TM_CCORR_NORMED
    res = cv2.matchTemplate(gray, templ, MATCH_METHOD, mask=mask)
    return res


def find_candidates_for_one_template(gray, tpl_pack):
    templ0 = tpl_pack["templ"]
    mask0 = tpl_pack["mask"]
    meta0 = tpl_pack["meta"]

    H, W = gray.shape[:2]
    candidates = []

    for s in SCALES:
        templ, mask, meta = resize_template_pack(templ0, mask0, meta0, s)
        th, tw = templ.shape[:2]

        if th >= H or tw >= W:
            continue

        res = template_match_with_mask(gray, templ, mask)

        # 取所有高于阈值的位置
        ys, xs = np.where(res >= SCORE_THRESH)
        for y, x in zip(ys, xs):
            score = float(res[y, x])

            # 模板左上角是 (x, y)
            # 由模板中 bbox_center_in_roi 反推出健康目标中心
            local_cx, local_cy = meta["bbox_center_in_roi"]
            pred_cx = x + local_cx
            pred_cy = y + local_cy

            bw, bh = meta["bbox_size"]
            pred_x1 = int(round(pred_cx - bw / 2))
            pred_y1 = int(round(pred_cy - bh / 2))
            pred_x2 = int(round(pred_cx + bw / 2))
            pred_y2 = int(round(pred_cy + bh / 2))

            pred_x1 = max(0, pred_x1)
            pred_y1 = max(0, pred_y1)
            pred_x2 = min(W, pred_x2)
            pred_y2 = min(H, pred_y2)

            if pred_x2 <= pred_x1 or pred_y2 <= pred_y1:
                continue

            candidates.append({
                "template_name": tpl_pack["name"],
                "scale": s,
                "score": score,
                "match_box": [int(x), int(y), int(x + tw), int(y + th)],
                "pred_box": [pred_x1, pred_y1, pred_x2, pred_y2]
            })

    return candidates


def edge_margin_ok(box, img_w, img_h, ratio=0.03):
    x1, y1, x2, y2 = box
    mx = img_w * ratio
    my = img_h * ratio
    return (x1 >= mx and y1 >= my and x2 <= img_w - mx and y2 <= img_h - my)


def rank_candidates(gray, cands):
    scored = []
    for c in cands:
        box = c["pred_box"]
        sharp = compute_sharpness(gray, box)

        # 简单归一化
        sharp_score = min(sharp / 300.0, 1.0)

        final = 0.8 * c["score"] + 0.2 * sharp_score
        c["sharpness"] = sharp
        c["final_score"] = final
        scored.append(c)

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored


def draw_results(img, cands):
    vis = img.copy()
    for i, c in enumerate(cands):
        x1, y1, x2, y2 = c["pred_box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{i+1}:{c['final_score']:.3f}"
        cv2.putText(vis, text, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return vis


def main():
    ensure_dirs()
    templates = load_templates()
    print(f"[INFO] loaded {len(templates)} templates")

    healthy_paths = sorted(Path(HEALTHY_IMG_DIR).glob("*.*"))

    for img_path in healthy_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gray = preprocess_gray(img)
        H, W = gray.shape[:2]

        all_cands = []
        for tpl in templates:
            cands = find_candidates_for_one_template(gray, tpl)
            all_cands.extend(cands)

        # 规则过滤：边缘安全距离
        all_cands = [
            c for c in all_cands
            if edge_margin_ok(c["pred_box"], W, H, EDGE_MARGIN_RATIO)
        ]

        # NMS
        all_cands = nms(all_cands, NMS_IOU_THRESH)

        # 排序
        all_cands = rank_candidates(gray, all_cands)[:TOPK]

        stem = img_path.stem

        # 保存 JSON
        json_out = os.path.join(OUT_JSON_DIR, stem + ".json")
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(all_cands, f, ensure_ascii=False, indent=2)

        # 保存可视化
        vis = draw_results(img, all_cands)
        vis_out = os.path.join(OUT_VIS_DIR, stem + ".jpg")
        cv2.imwrite(vis_out, vis)

        print(f"[DONE] {img_path.name}: {len(all_cands)} candidates")

    print("[ALL DONE]")


if __name__ == "__main__":
    main()
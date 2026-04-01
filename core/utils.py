# core/utils.py - 完整稳定版（支持画框+保存，兼容所有场景）
import cv2
import os
import numpy as np
from datetime import datetime


def _get_class_color(label: str):
    """根据类别名生成稳定且区分度较高的颜色（BGR）。"""
    palette = [
        (56, 56, 255),
        (151, 157, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (23, 204, 146),
        (134, 219, 61),
        (52, 147, 26),
        (187, 212, 0),
        (168, 153, 44),
        (255, 194, 0),
        (147, 69, 52),
        (255, 115, 100),
        (236, 24, 0),
        (255, 56, 132),
        (133, 0, 82),
        (255, 56, 203),
        (200, 149, 255),
        (199, 55, 255),
    ]
    color_index = sum(ord(ch) for ch in str(label)) % len(palette)
    return palette[color_index]


def _get_draw_style(image_shape):
    """根据图片尺寸动态计算框线、字体、边距等绘制参数。"""
    height, width = image_shape[:2]
    scale_base = max(height, width)
    line_thickness = max(2, int(round(scale_base / 500)))
    font_scale = max(0.6, scale_base / 1200)
    font_thickness = max(1, line_thickness - 1)
    padding = max(4, line_thickness + 2)
    return line_thickness, font_scale, font_thickness, padding


def _get_text_color(background_color):
    """根据背景色亮度选择黑/白文字，提升可读性。"""
    b, g, r = background_color
    brightness = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if brightness > 160 else (255, 255, 255)

def draw_boxes(image_path, detect_result):
    """
    为图片绘制检测框（核心函数）
    :param image_path: 原图的绝对路径（支持中文路径）
    :param detect_result: 检测结果字典，格式要求：
                          {
                              "detections": [{"label": "类别名", "conf": 置信度, "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}],
                              "items": 同上（兼容两种字段名）
                          }
    :return: 绘制完成的图片数组（cv2格式），绘制失败返回原图数组
    """
    # 1. 读取图片（兼容中文路径，解决90%的图片读取问题）
    try:
        # 方式1：通过numpy读取二进制，再解码（兼容中文路径）
        img_buffer = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        if img is None:
            # 方式2：兜底（常规读取，兼容非中文路径）
            img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"图片读取失败：路径错误或格式不支持 → {image_path}")
    except Exception as e:
        print(f"❌ 读取图片出错：{e}")
        raise e

    # 2. 提取检测结果（兼容detections/items两种字段名）
    detections = detect_result.get("detections", detect_result.get("items", []))
    if not detections:
        print("ℹ️ 无检测结果，返回原图")
        return img

    line_thickness, font_scale, font_thickness, padding = _get_draw_style(img.shape)

    # 3. 遍历检测结果绘制框和标签
    for idx, item in enumerate(detections):
        # 提取基础信息
        label = item.get("label", f"未知目标{idx+1}")
        conf = item.get("conf", 0.0)
        bbox = item.get("bbox", {})

        # 兼容多种坐标字段名（x1/y1/x2/y2 或 xmin/ymin/xmax/ymax）
        x1 = int(bbox.get("x1", bbox.get("xmin", 0)))
        y1 = int(bbox.get("y1", bbox.get("ymin", 0)))
        x2 = int(bbox.get("x2", bbox.get("xmax", 0)))
        y2 = int(bbox.get("y2", bbox.get("ymax", 0)))

        # 坐标裁剪到图像范围内，避免越界
        x1 = max(0, min(x1, img.shape[1] - 1))
        y1 = max(0, min(y1, img.shape[0] - 1))
        x2 = max(0, min(x2, img.shape[1] - 1))
        y2 = max(0, min(y2, img.shape[0] - 1))

        # 过滤无效坐标（避免绘制错误）
        if x1 >= x2 or y1 >= y2:
            print(f"⚠️ 跳过无效坐标：目标{idx+1} → x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            continue

        box_color = _get_class_color(label)
        text_color = _get_text_color(box_color)

        # 4. 绘制检测框（按类别区分颜色，线宽随图片尺寸自适应）
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, line_thickness, lineType=cv2.LINE_AA)

        # 5. 绘制标签背景和文字（显示类别名 + 两位小数置信度）
        label_text = f"{label} {float(conf):.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )

        text_box_top = y1 - text_height - baseline - padding * 2
        if text_box_top < 0:
            text_box_top = y1
            text_box_bottom = min(img.shape[0] - 1, y1 + text_height + baseline + padding * 2)
            text_y = text_box_bottom - baseline - padding
        else:
            text_box_bottom = y1
            text_y = y1 - baseline - padding

        cv2.rectangle(
            img,
            (x1, text_box_top),
            (min(img.shape[1] - 1, x1 + text_width + padding * 2), text_box_bottom),
            box_color,
            -1  # 填充背景
        )

        # 6. 绘制标签文字（白色，对比明显）
        cv2.putText(
            img,
            label_text,
            (x1 + padding, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    print(f"✅ 绘图完成：共绘制{len(detections)}个有效检测框")
    return img

def save_result(drawn_img, save_dir=None, prefix="result"):
    """
    保存绘制好的图片（支持中文路径）
    :param drawn_img: 绘制完成的图片数组（cv2格式）
    :param save_dir: 保存目录（默认：B_ROOT/assets/output）
    :param prefix: 文件名前缀（默认：result）
    :return: 保存后的图片绝对路径
    """
    # 1. 确定保存目录（优先使用传入的目录，无则自动创建）
    if save_dir is None:
        # 自动定位到B代码根目录下的assets/output
        current_dir = os.path.dirname(os.path.abspath(__file__))  # utils.py所在目录
        b_root = os.path.dirname(os.path.dirname(current_dir))    # 上两级 = B_ROOT
        save_dir = os.path.join(b_root, "assets", "output")
    os.makedirs(save_dir, exist_ok=True)

    # 2. 生成唯一文件名（避免重复）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.jpg"
    save_path = os.path.join(save_dir, filename)

    # 3. 保存图片（兼容中文路径）
    try:
        # cv2.imwrite不支持中文路径，改用imencode+tofile
        _, img_buffer = cv2.imencode(".jpg", drawn_img)
        img_buffer.tofile(save_path)

        # 验证保存是否成功
        if not os.path.exists(save_path):
            raise Exception("图片保存后未找到文件")

        print(f"✅ 图片保存成功：{save_path}")
        return save_path
    except Exception as e:
        print(f"❌ 保存图片失败：{e}")
        # 兜底：尝试用常规方式保存（非中文路径可用）
        cv2.imwrite(save_path, drawn_img)
        return save_path

# ========== 可选：测试函数（单独运行utils.py验证功能） ==========
if __name__ == "__main__":
    # 测试用例：替换为你的测试图片路径
    test_image_path = "C:/测试图片.jpg"  # 支持中文路径
    # 模拟检测结果（和你的detect_result格式一致）
    test_detect_result = {
        "detections": [
            {"label": "person", "conf": 0.85, "bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 300}},
            {"label": "car", "conf": 0.92, "bbox": {"xmin": 300, "ymin": 200, "xmax": 400, "ymax": 300}}  # 兼容xmin格式
        ]
    }

    # 测试绘图+保存
    try:
        drawn_img = draw_boxes(test_image_path, test_detect_result)
        save_result(drawn_img)
        print("🎉 测试成功！")
    except Exception as e:
        print(f"❌ 测试失败：{e}")
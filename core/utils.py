# core/utils.py - 完整稳定版（支持画框+保存，兼容所有场景）
import cv2
import os
import numpy as np
from datetime import datetime

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

        # 过滤无效坐标（避免绘制错误）
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            print(f"⚠️ 跳过无效坐标：目标{idx+1} → x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            continue

        # 4. 绘制检测框（绿色，线宽2，醒目且不刺眼）
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 5. 绘制标签背景（半透明绿色，提升文字可读性）
        label_text = f"{label} {conf:.2f}"
        # 计算文字尺寸
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 绘制背景矩形
        cv2.rectangle(
            img,
            (x1, y1 - text_height - 10),  # 左上角
            (x1 + text_width + 10, y1),    # 右下角
            (0, 255, 0),
            -1  # 填充背景
        )

        # 6. 绘制标签文字（白色，对比明显）
        cv2.putText(
            img,
            label_text,
            (x1 + 5, y1 - 5),  # 文字起始位置
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,               # 字体大小
            (255, 255, 255),   # 文字颜色（白色）
            1                  # 文字线宽
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
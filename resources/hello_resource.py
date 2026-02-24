"""
YOLO 检测资源模块
提供检测历史、模型信息等资源访问
"""
import os
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

from resources import YA_MCPServer_Resource


@YA_MCPServer_Resource(
    "yolo://history/list",  # 资源 URI
    name="detection_history",  # 资源 ID
    title="Detection History",  # 资源标题
    description="返回最近检测的图片列表",  # 资源描述
)
def get_detection_history() -> List[Dict[str, Any]]:
    """
    返回最近检测的图片列表。

    Returns:
        List[Dict[str, Any]]: 包含检测历史记录的列表。

    Example:
        [
            {
                "filename": "detect_20240101_120000.jpg",
                "path": "assets/output/detect_20240101_120000.jpg",
                "time": "2024-01-01 12:00:00",
                "size": "1.2 MB"
            }
        ]
    """
    output_dir = Path("assets/output")
    
    if not output_dir.exists():
        return []
    
    # 获取所有图片文件
    image_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
    
    # 按修改时间排序，取最近10张
    image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    image_files = image_files[:10]
    
    history = []
    for img_path in image_files:
        stat = img_path.stat()
        history.append({
            "filename": img_path.name,
            "path": str(img_path),
            "time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "size": f"{stat.st_size / 1024 / 1024:.2f} MB" if stat.st_size > 1024*1024 else f"{stat.st_size / 1024:.2f} KB"
        })
    
    return history


@YA_MCPServer_Resource(
    "yolo://models/info",  # 资源 URI
    name="model_info",  # 资源 ID
    title="YOLO Model Information",  # 资源标题
    description="返回可用 YOLO 模型的信息",  # 资源描述
)
def get_model_info() -> Dict[str, Any]:
    """
    返回可用 YOLO 模型的信息。

    Returns:
        Dict[str, Any]: 包含模型信息的字典。

    Example:
        {
            "v5": {
                "available": true,
                "path": "models/yolov5s.pt",
                "description": "YOLOv5s - 轻量快速"
            }
        }
    """
    from modules.YA_Common.utils.config import get_config
    
    v5_path = get_config("yolo.v5_path", "models/yolov5s.pt")
    v8_path = get_config("yolo.v8_path", "models/yolov8n.pt")
    
    return {
        "v5": {
            "available": os.path.exists(v5_path),
            "path": v5_path,
            "description": "YOLOv5s - 轻量快速，适合实时检测"
        },
        "v8": {
            "available": os.path.exists(v8_path),
            "path": v8_path,
            "description": "YOLOv8n - 精度更高，适合复杂场景"
        }
    }


@YA_MCPServer_Resource(
    "yolo://stats/summary",  # 资源 URI
    name="detection_stats",  # 资源 ID
    title="Detection Statistics",  # 资源标题
    description="返回检测统计摘要",  # 资源描述
)
def get_detection_stats() -> Dict[str, Any]:
    """
    返回检测统计摘要。

    Returns:
        Dict[str, Any]: 包含统计信息的字典。

    Example:
        {
            "total_detections": 42,
            "today_detections": 5,
            "avg_objects_per_image": 3.2
        }
    """
    output_dir = Path("assets/output")
    input_dir = Path("assets/input")
    
    # 确保目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计 output 目录
    output_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
    
    # 统计今天的文件
    today = datetime.now().strftime("%Y%m%d")
    today_files = [f for f in output_files if today in f.name]
    
    return {
        "total_detections": len(output_files),
        "today_detections": len(today_files),
        "input_files_available": len(list(input_dir.glob("*"))),
        "output_directory": str(output_dir),
        "input_directory": str(input_dir)
    }


@YA_MCPServer_Resource(
    "yolo://result/{filename}",  # 资源模板 URI
    name="get_detection_result",  # 资源 ID
    title="Get Detection Result",  # 资源标题
    description="返回指定检测结果图片的信息",  # 资源描述
)
def get_detection_result(filename: str) -> Dict[str, Any]:
    """
    返回指定检测结果图片的信息。

    Args:
        filename (str): 结果图片的文件名。

    Returns:
        Dict[str, Any]: 包含图片信息的字典。

    Example:
        {
            "success": true,
            "path": "assets/output/detect_20240101_120000.jpg",
            "size": "1.2 MB",
            "time": "2024-01-01 12:00:00"
        }
    """
    output_dir = Path("assets/output")
    file_path = output_dir / filename
    
    if not file_path.exists():
        return {
            "success": False,
            "error": f"File not found: {filename}"
        }
    
    stat = file_path.stat()
    return {
        "success": True,
        "filename": filename,
        "path": str(file_path),
        "size": f"{stat.st_size / 1024:.2f} KB",
        "time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "modified": stat.st_mtime
    }
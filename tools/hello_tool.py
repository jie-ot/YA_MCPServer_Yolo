"""
YOLO 目标检测工具
"""
from typing import Any, Dict, List, Optional

from tools import YA_MCPServer_Tool


@YA_MCPServer_Tool(
    name="detect_image",
    title="YOLO 目标检测",
    description="对图片进行目标检测，返回检测到的物体列表",
)
async def detect_image(
    image_path: str, 
    model_type: str = "v5",
    conf_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """对图片进行目标检测。

    Args:
        image_path (str): 图片路径，可以是 assets/input 下的文件名或完整路径。
        model_type (str, optional): 模型类型，"v5" 或 "v8"。默认为 "v5"。
        conf_threshold (float, optional): 置信度阈值，低于此值的检测结果将被过滤。
            默认从配置文件读取。

    Returns:
        Dict[str, Any]: 包含检测结果的字典。

    Example:
        {
            "success": True,
            "total_objects": 3,
            "objects": [
                {
                    "label": "person",
                    "confidence": 0.95,
                    "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 500}
                }
            ]
        }
    """
    try:
        import os
        from pathlib import Path
        from core.inference import YoloEngine, DetectionResult
        from modules.YA_Common.utils.config import get_config
        from modules.YA_Common.utils.logger import get_logger
    except ImportError as e:
        raise RuntimeError(f"无法导入必要模块: {e}")

    logger = get_logger("YA_MCPServer_Tools.YOLO")
    logger.info(f"开始检测图片: {image_path}, 模型: {model_type}")
    
    try:
        # 处理图片路径
        if not os.path.isabs(image_path):
            input_dir = Path("assets/input")
            input_dir.mkdir(parents=True, exist_ok=True)
            full_path = input_dir / image_path
        else:
            full_path = Path(image_path)
        
        if not full_path.exists():
            return {
                "success": False,
                "error": f"图片不存在: {full_path}"
            }
        
        # 获取置信度阈值
        if conf_threshold is None:
            conf_threshold = get_config("yolo.conf_threshold", 0.25)
        
        # 执行检测
        engine = YoloEngine()
        if not engine.models:
            await engine.load_models()
            
        result: DetectionResult = await engine.detect(
            str(full_path), 
            model_type=model_type
        )
        
        # 根据阈值过滤
        if conf_threshold > 0:
            result.items = [
                item for item in result.items 
                if item.conf >= conf_threshold
            ]
            result.count = len(result.items)
        
        # 构建返回数据
        response = {
            "success": True,
            "model": result.model,
            "total_objects": result.count,
            "objects": [
                {
                    "label": item.label,
                    "confidence": round(item.conf, 3),
                    "bbox": {
                        "x1": item.bbox.x1,
                        "y1": item.bbox.y1,
                        "x2": item.bbox.x2,
                        "y2": item.bbox.y2
                    }
                }
                for item in result.items
            ]
        }
        
        # 添加统计信息
        if result.count > 0:
            label_counts = {}
            confidences = []
            for item in result.items:
                label_counts[item.label] = label_counts.get(item.label, 0) + 1
                confidences.append(item.conf)
            
            response["statistics"] = {
                "label_counts": label_counts,
                "avg_confidence": round(sum(confidences) / len(confidences), 3),
                "max_confidence": round(max(confidences), 3),
                "min_confidence": round(min(confidences), 3)
            }
        
        logger.info(f"检测完成，共检测到 {result.count} 个目标")
        return response
        
    except Exception as e:
        logger.error(f"检测失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@YA_MCPServer_Tool(
    name="list_yolo_models",
    title="List YOLO Models",
    description="列出可用的 YOLO 模型及其状态",
)
async def list_yolo_models() -> Dict[str, Any]:
    """列出可用的 YOLO 模型。

    Returns:
        Dict[str, Any]: 包含模型信息的字典。

    Example:
        {
            "success": True,
            "models": {
                "v5": {
                    "available": True,
                    "path": "models/yolov5s.pt",
                    "description": "YOLOv5s - 轻量快速"
                }
            },
            "default": "v5"
        }
    """
    try:
        import os
        from modules.YA_Common.utils.config import get_config
    except ImportError as e:
        raise RuntimeError(f"无法导入配置模块: {e}")
    
    try:
        v5_path = get_config("yolo.v5_path", "models/yolov5s.pt")
        v8_path = get_config("yolo.v8_path", "models/yolov8n.pt")
        
        return {
            "success": True,
            "models": {
                "v5": {
                    "available": os.path.exists(v5_path),
                    "path": v5_path,
                    "description": "YOLOv5s - 轻量快速"
                },
                "v8": {
                    "available": os.path.exists(v8_path),
                    "path": v8_path,
                    "description": "YOLOv8n - 精度更高"
                }
            },
            "default": get_config("yolo.default_model", "v5")
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@YA_MCPServer_Tool(
    name="preload_yolo_models",
    title="Preload YOLO Models",
    description="预加载 YOLO 模型到内存，加快首次检测速度",
)
async def preload_yolo_models() -> Dict[str, Any]:
    """预加载 YOLO 模型。

    Returns:
        Dict[str, Any]: 包含加载结果的字典。

    Example:
        {
            "success": True,
            "message": "模型加载完成",
            "loaded_models": ["v5"]
        }
    """
    try:
        from core.inference import YoloEngine
        from modules.YA_Common.utils.logger import get_logger
    except ImportError as e:
        raise RuntimeError(f"无法导入必要模块: {e}")
    
    logger = get_logger("YA_MCPServer_Tools.YOLO")
    
    try:
        engine = YoloEngine()
        await engine.load_models()
        
        return {
            "success": True,
            "message": "模型加载完成",
            "loaded_models": list(engine.models.keys())
        }
    except Exception as e:
        logger.error(f"模型预加载失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@YA_MCPServer_Tool(
    name="get_yolo_stats",
    title="Get YOLO Statistics",
    description="获取 YOLO 检测的统计信息",
)
async def get_yolo_stats() -> Dict[str, Any]:
    """获取检测统计信息。

    Returns:
        Dict[str, Any]: 包含统计信息的字典。

    Example:
        {
            "success": True,
            "stats": {
                "total_detections": 10,
                "available_models": ["v5", "v8"],
                "input_dir": "assets/input",
                "output_dir": "assets/output"
            }
        }
    """
    try:
        from pathlib import Path
    except ImportError as e:
        raise RuntimeError(f"无法导入必要模块: {e}")
    
    try:
        input_dir = Path("assets/input")
        output_dir = Path("assets/output")
        
        # 确保目录存在
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计 output 目录下的图片
        output_files = []
        if output_dir.exists():
            output_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
        
        return {
            "success": True,
            "stats": {
                "total_detections": len(output_files),
                "available_models": ["v5", "v8"],
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "input_files": [f.name for f in input_dir.glob("*") if f.is_file()]
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
"""
YOLO Inference Engine 核心推理模块
负责加载模型、执行推理、处理数据格式。
"""
import os
import sys
import torch
import pathlib
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from ultralytics import YOLO
from modules.YA_Common.utils.config import get_config

# 配置日志
logger = logging.getLogger("core.inference")

# 路径兼容修复
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# 定义数据模型 ---
class BBox(BaseModel):
    x1: int = Field(..., description="左上角 X 坐标")
    y1: int = Field(..., description="左上角 Y 坐标")
    x2: int = Field(..., description="右下角 X 坐标")
    y2: int = Field(..., description="右下角 Y 坐标")

class DetectionItem(BaseModel):
    label: str = Field(..., description="类别名称")
    conf: float = Field(..., description="置信度 (0-1)")
    bbox: BBox = Field(..., description="边界框")

class DetectionResult(BaseModel):
    model: str = Field(..., description="使用的模型类型 (v5/v8)")
    count: int = Field(..., description="检测到的物体总数")
    items: List[DetectionItem] = Field(..., description="检测对象列表")

# 核心推理类
class YoloEngine:
    """
    YOLO 目标检测引擎
    单例模式，支持 YOLOv5 和 YOLOv8 模型加载与推理。
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YoloEngine, cls).__new__(cls)
            cls._instance.models = {}
        return cls._instance

    async def load_models(self) -> None:
        if self.models:
            return

        v5_path = get_config("yolo.v5_path", "models/yolov5s.pt")
        v8_path = get_config("yolo.v8_path", "models/yolov8n.pt")

        # 加载 YOLOv5
        try:
            yolov5_repo = os.path.join(os.path.dirname(__file__), "yolov5")
            if yolov5_repo not in sys.path:
                sys.path.append(yolov5_repo)
            
            self.models['v5'] = torch.hub.load(
                yolov5_repo, 'custom', path=v5_path, source='local'
            )
            logger.info(f"YOLOv5 loaded from {v5_path}")
        except Exception as e:
            logger.error(f"Error loading YOLOv5: {e}")

        # 加载 YOLOv8
        try:
            self.models['v8'] = YOLO(v8_path)
            logger.info(f"YOLOv8 loaded from {v8_path}")
        except Exception as e:
            logger.error(f"Error loading YOLOv8: {e}")

    async def detect(self, image_path: str, model_type: str = 'v5') -> DetectionResult:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not self.models:
            await self.load_models()
            
        if model_type not in self.models:
            raise ValueError(f"Model '{model_type}' is not loaded or supported.")

        try:
            items = []
            
            if model_type == 'v5':
                model = self.models['v5']
                results = model(image_path)
                
                # 解析 v5 结果
                for *xyxy, conf, cls in results.xyxy[0]:
                    items.append(DetectionItem(
                        label=results.names[int(cls)],
                        conf=float(conf),
                        bbox=BBox(
                            x1=int(xyxy[0]), y1=int(xyxy[1]),
                            x2=int(xyxy[2]), y2=int(xyxy[3])
                        )
                    ))
                    
            elif model_type == 'v8':
                model = self.models['v8']
                results = model(image_path)
                
                # 解析 v8 结果
                for r in results:
                    for box in r.boxes:
                        coords = box.xyxy[0].tolist()
                        items.append(DetectionItem(
                            label=model.names[int(box.cls)],
                            conf=float(box.conf),
                            bbox=BBox(
                                x1=int(coords[0]), y1=int(coords[1]),
                                x2=int(coords[2]), y2=int(coords[3])
                            )
                        ))

            return DetectionResult(
                model=model_type,
                count=len(items),
                items=items
            )

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}")
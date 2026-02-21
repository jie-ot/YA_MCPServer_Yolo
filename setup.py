import os
import requests
from modules.YA_Common.utils.logger import get_logger

logger = get_logger("setup")

# 模型下载配置
MODELS = {
    "yolov5s.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
}

def setup():
    """Setup your environment and dependencies here."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 创建必要的目录
        dirs = [
            os.path.join(base_dir, "assets", "input"),
            os.path.join(base_dir, "assets", "output"),
            os.path.join(base_dir, "models"),
        ]
        
        for d in dirs:
            os.makedirs(d, exist_ok=True)
            logger.debug(f"Checked directory: {d}") # 可以考虑注释掉

        # 下载模型
        for name, url in MODELS.items():
            path = os.path.join(base_dir, "models", name)
            if not os.path.exists(path):
                logger.info(f"Downloading {name}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Downloaded {name} successfully.")
            else:
                logger.info(f"Model {name} exists locally. Skipping.")

        logger.info("Setup complete.")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise e

if __name__ == "__main__":
    setup()
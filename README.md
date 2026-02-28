## YA_MCPServer_Yolo

基于 MCP 协议的 YOLO 目标检测智能体服务器，支持 YOLOv5 与 YOLOv8 双引擎，提供图像检测、结果分析与可视化功能。

### 组员信息

| 姓名 | 学号 | 分工 |
| :--: | :--: | :--: |
| 赵冠杰 | U202414739 | 核心算法与环境配置 (Core & Setup) |
| 王子舜 | U202414921 | MCP 协议实现 (Tools, Resources, Prompts) |
| 陈思宇 | U202414897 | 客户端交互与可视化 (Notebook & Utils) |

### Tool 列表

| 工具名称 | 功能描述 | 输入 | 输出 | 备注 |
| :------: | :------: | :--: | :--: | :--: |
| `detect_image` | 对图片进行目标检测，返回检测到的物体列表 | `image_path` (图片路径), `model_type` (模型类型), `conf_threshold` (置信度阈值) | `Dict` (包含检测成功状态、物体总数及具体坐标和置信度的 JSON) | 核心检测工具 |
| `list_yolo_models` | 列出可用的 YOLO 模型及其状态 | 无 | `Dict` (模型列表及可用状态) | |
| `preload_yolo_models` | 预加载 YOLO 模型到内存，加快首次检测速度 | `model_type` (可选，指定预加载模型) | `Dict` (加载结果状态) | 优化性能 |
| `get_yolo_stats` | 获取 YOLO 检测的统计信息 | 无 | `Dict` (历史检测总数、平均耗时等) | |

### Resource 列表

| 资源名称 | 功能描述 | 输入 | 输出 | 备注 |
| :------: | :------: | :--: | :--: | :--: |
| `yolo://history/list` | 返回最近检测的图片列表 | 无 | `List[Dict]` (包含文件名、路径、时间、大小的列表) | 默认返回最近10条 |
| `yolo://models/info` | 返回可用 YOLO 模型的信息 | 无 | `Dict` (模型路径、描述及可用状态) | |
| `yolo://stats/summary` | 返回检测统计摘要 | 无 | `Dict` (统计数据) | |
| `yolo://history/{filename}` | 返回指定检测结果图片的信息 | `filename` (文件名) | `Dict` (单张图片的详细信息) | |

### Prompts 列表

| 指令名称 | 功能描述 | 输入 | 输出 | 备注 |
| :------: | :------: | :--: | :--: | :--: |
| `analyze_security` | 分析检测结果中的安全隐患 | `detection_result` (检测结果 JSON) | `str` (引导 LLM 进行安防评估的提示词) | 自动识别刀具等危险品 |
| `count_objects` | 统计各类别物体的数量 | `detection_result` (检测结果 JSON) | `str` (引导 LLM 进行数量盘点的提示词) | 适用于仓储场景 |
| `quality_assessment` | 评估检测结果的质量 | `detection_result` (检测结果 JSON) | `str` (引导 LLM 评估置信度分布的提示词) | |
| `scene_description` | 根据检测结果描述场景 | `detection_result` (检测结果 JSON) | `str` (引导 LLM 生成场景自然语言描述的提示词) | |

### 项目结构

- `core`: 核心业务逻辑层。
  - `inference.py`: 封装了 `YoloEngine` 单例类，负责 YOLO 模型的异步加载与推理，统一数据输出格式。
  - `utils.py`: 图像处理工具，负责解析坐标数据并在原图上绘制边界框及保存结果。
  - `yolov5/`: 克隆的 YOLOv5 官方仓库，作为本地依赖供引擎调用。
- `tools`: MCP Tool 实现层，定义了供大模型调用的工具接口（如 `detect_image`）。
- `resources`: MCP Resource 实现层，暴露本地检测历史和模型状态供大模型读取。
- `prompts`: MCP Prompt 实现层，定义了将检测数据转化为自然语言分析报告的模板。
- `assets`: 运行时资源目录。包含 `input/` (用户上传图片)、`output/` (画框后的结果图) 。
- `models`: 模型权重仓库，存放 `yolov5s.pt` 和 `yolov8n.pt` 文件。
- `config.yaml`: 配置文件，添加了 YOLO 模型的本地路径、默认置信度阈值 (`conf_thres`) 以及可视化输出的相关配置。
- `setup.py`: 环境初始化脚本，用于自动创建必要的目录结构并从远程下载缺失的模型权重文件。

### 其他需要说明的情况

- 使用了 PyTorch 深度学习框架
- 使用了机器学习、深度学习模型（YOLOv5 & YOLOv8）

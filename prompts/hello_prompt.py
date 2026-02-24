"""
YOLO 检测提示词模块
提供检测结果分析、场景评估等提示词模板
"""
from typing import Any, Dict, List

from prompts import YA_MCPServer_Prompt


@YA_MCPServer_Prompt(
    name="analyze_security",
    title="Security Analysis",
    description="分析检测结果中的安全隐患",
)
async def analyze_security(detection_result: Dict[str, Any]) -> str:
    """分析检测结果中的安全隐患。

    Args:
        detection_result (Dict[str, Any]): 检测结果数据，来自 detect_image 工具。

    Returns:
        str: 安全分析提示词。

    Example:
        {
            "total_objects": 3,
            "objects": [
                {"label": "person", "confidence": 0.95},
                {"label": "knife", "confidence": 0.87}
            ]
        }
    """
    if not detection_result.get("success", False):
        return f"无法分析：{detection_result.get('error', '未知错误')}"
    
    objects = detection_result.get("objects", [])
    total = detection_result.get("total_objects", 0)
    
    # 提取标签列表
    labels = [obj["label"] for obj in objects]
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # 检测危险物品
    dangerous_items = ["knife", "gun", "weapon", "scissors", "bottle"]
    detected_dangers = [label for label in labels if label in dangerous_items]
    
    prompt = f"""请基于以下目标检测结果进行安全评估：

检测统计：
- 共检测到 {total} 个物体
- 物体类别分布：{', '.join([f'{k} x{v}' for k, v in label_counts.items()])}

检测到的具体物体：
{chr(10).join([f'  - {obj["label"]} (置信度: {obj["confidence"]})' for obj in objects])}

请分析：
1. 场景的整体安全性如何？
2. 是否存在安全隐患？{f'（检测到危险物品：{", ".join(detected_dangers)}）' if detected_dangers else ''}
3. 如果有安全风险，建议采取什么措施？
4. 如果没有明显风险，请描述这是一个什么样的场景。

请用中文回答，给出详细的分析建议。"""
    
    return prompt


@YA_MCPServer_Prompt(
    name="count_objects",
    title="Object Counting",
    description="统计各类别物体的数量",
)
async def count_objects(detection_result: Dict[str, Any]) -> str:
    """统计各类别物体的数量。

    Args:
        detection_result (Dict[str, Any]): 检测结果数据。

    Returns:
        str: 物体统计提示词。
    """
    if not detection_result.get("success", False):
        return f"无法统计：{detection_result.get('error', '未知错误')}"
    
    objects = detection_result.get("objects", [])
    total = detection_result.get("total_objects", 0)
    
    # 统计各类别数量
    counts = {}
    for obj in objects:
        label = obj["label"]
        counts[label] = counts.get(label, 0) + 1
    
    # 按数量排序
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    prompt = f"""请帮我统计和分析检测结果：

总检测数量：{total} 个物体

各类别数量统计：
{chr(10).join([f'- {label}: {count} 个' for label, count in sorted_counts])}

请回答：
1. 画面中最多的是什么物体？有多少个？
2. 这个分布是否合理？为什么？
3. 如果需要，建议如何进一步分析这些数据？

请用中文简洁回答。"""
    
    return prompt


@YA_MCPServer_Prompt(
    name="quality_assessment",
    title="Detection Quality Assessment",
    description="评估检测结果的质量",
)
async def quality_assessment(detection_result: Dict[str, Any]) -> str:
    """评估检测结果的质量。

    Args:
        detection_result (Dict[str, Any]): 检测结果数据。

    Returns:
        str: 质量评估提示词。
    """
    if not detection_result.get("success", False):
        return f"无法评估：{detection_result.get('error', '未知错误')}"
    
    objects = detection_result.get("objects", [])
    model = detection_result.get("model", "unknown")
    
    if not objects:
        return "检测结果为空，请检查：\n1. 图片中是否有目标物体？\n2. 置信度阈值是否设置过高？\n3. 模型是否正确加载？"
    
    confidences = [obj["confidence"] for obj in objects]
    avg_conf = sum(confidences) / len(confidences)
    max_conf = max(confidences)
    min_conf = min(confidences)
    
    prompt = f"""请评估这次目标检测的质量：

检测信息：
- 使用模型：YOLO{model}
- 检测到 {len(objects)} 个物体
- 平均置信度：{avg_conf:.3f}
- 最高置信度：{max_conf:.3f}
- 最低置信度：{min_conf:.3f}

物体列表（按置信度排序）：
{chr(10).join([f'  {i+1}. {obj["label"]} ({obj["confidence"]:.3f})' for i, obj in enumerate(sorted(objects, key=lambda x: x["confidence"], reverse=True))])}

请评估：
1. 这次检测的总体置信度水平如何？
2. 是否有置信度特别低的检测？可能的原因是什么？
3. 模型选择（YOLO{model}）是否合适？
4. 建议如何优化检测效果？

请用中文回答。"""
    
    return prompt


@YA_MCPServer_Prompt(
    name="scene_description",
    title="Scene Description",
    description="根据检测结果描述场景",
)
async def scene_description(detection_result: Dict[str, Any]) -> str:
    """根据检测结果描述场景。

    Args:
        detection_result (Dict[str, Any]): 检测结果数据。

    Returns:
        str: 场景描述提示词。
    """
    if not detection_result.get("success", False):
        return f"无法描述：{detection_result.get('error', '未知错误')}"
    
    objects = detection_result.get("objects", [])
    
    if not objects:
        return "未检测到任何物体，可能是一个空旷场景或者图片质量问题。"
    
    labels = [obj["label"] for obj in objects]
    unique_labels = list(set(labels))
    
    prompt = f"""请根据检测到的物体描述这个场景：

检测到的物体类别：{', '.join(unique_labels)}
总物体数量：{len(objects)} 个

请发挥想象力，描述一个包含这些物体的场景。要求：
1. 描述要生动具体
2. 体现出物体之间的空间关系
3. 推测可能的场景氛围或活动
4. 控制在100字左右

场景描述："""
    
    return prompt
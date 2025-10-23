# src/detection_module.py
import json
import logging
from src.api_client import global_api_client
from src.prompts import (
    DETECTION_DIRECT_ASK_SYSTEM_PROMPT, get_detection_direct_ask_user_prompt,
    DETECTION_COT_SYSTEM_PROMPT, get_detection_cot_user_prompt,
    DETECTION_FEW_SHOT_SYSTEM_PROMPT, get_detection_few_shot_user_prompt
)
from config import (
    DETECTOR_MODEL_NAME,
    REQUEST_TEMPERATURE,
    DetectionStrategy
)

logger = logging.getLogger(__name__)


def _parse_llm_json_response(response_content: str) -> dict:
    """
    一个健壮的内部工具函数，用于解析LLM可能返回的JSON。
    它尝试处理 LLM 可能返回的被```json ... ```包裹的字符串。
    """
    try:
        # 尝试直接解析
        return json.loads(response_content)
    except json.JSONDecodeError:
        print("  [Detect Module] JSON 解析失败，尝试清理...")
        # 尝试从Markdown代码块中提取
        if "```json" in response_content:
            try:
                cleaned_content = response_content.split("```json")[1].split("```")[0].strip()
                return json.loads(cleaned_content)
            except (IndexError, json.JSONDecodeError) as e:
                print(f"  [Detect Module] 清理后JSON解析仍失败: {e}")
                pass

        # 如果都失败了
        print("  [Detect Module] 错误：无法解析评估器返回的JSON。")
        return {
            "is_hallucination": "error",
            "explanation": f"JSON解析失败。模型原始输出: {response_content}"
        }


def _call_detector_llm(system_prompt: str, user_prompt: str) -> dict:
    """
    调用评估器 LLM 并解析其 JSON 输出的内部函数。
    """
    if global_api_client is None:
        raise ValueError("API Client not initialized.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = global_api_client.create_chat_completion(
            model=DETECTOR_MODEL_NAME,
            messages=messages,
            temperature=REQUEST_TEMPERATURE,
            # 仅在 CoT 和 Direct Ask 时强制使用 JSON 模式
            # Few-shot 的 prompt 太长，有时在 JSON 模式下表现不佳
            response_format={"type": "json_object"} if "示例" not in user_prompt else None
        )

        response_content = response.choices[0].message.content
        return _parse_llm_json_response(response_content)

    except Exception as e:
        print(f"  [Detect Module] 调用评估器LLM时出错: {e}")
        logger.error(f"调用DeepSeek API失败 (检测阶段): {e}", exc_info=True)
        return {
            "is_hallucination": "error",
            "explanation": str(e)
        }


def detect_hallucination(context: str, answer: str, strategy: str) -> dict:
    """
    幻觉检测的核心函数。
    根据传入的 'strategy' 参数，选择不同的 Prompt 和逻辑来执行检测。
    """
    print(f"  [Detect Module] 正在执行幻觉检测 (策略: {strategy})...")

    # 预检：如果模型自己说“无法回答”，则它没有产生幻觉
    if "无法回答" in answer or "未提及" in answer:
        print("  [Detect Module] 检测到'无法回答'，判定为 [无幻觉]。")
        return {
            "strategy_used": strategy,
            "is_hallucination": False,
            "explanation": "模型遵守了指令，正确地指出上下文中没有答案，因此没有产生幻觉。"
        }

    # 根据策略选择 Prompts
    if strategy == DetectionStrategy.DIRECT_ASK:
        system_prompt = DETECTION_DIRECT_ASK_SYSTEM_PROMPT
        user_prompt = get_detection_direct_ask_user_prompt(context, answer)

    elif strategy == DetectionStrategy.CHAIN_OF_THOUGHT:
        system_prompt = DETECTION_COT_SYSTEM_PROMPT
        user_prompt = get_detection_cot_user_prompt(context, answer)

    elif strategy == DetectionStrategy.FEW_SHOT:
        system_prompt = DETECTION_FEW_SHOT_SYSTEM_PROMPT
        user_prompt = get_detection_few_shot_user_prompt(context, answer)

    else:
        print(f"  [Detect Module] 错误：未知的检测策略 '{strategy}'。")
        return {
            "strategy_used": strategy,
            "is_hallucination": "error",
            "explanation": f"未知的检测策略: {strategy}"
        }

    # 调用 LLM 进行评估
    result = _call_detector_llm(system_prompt, user_prompt)

    # 统一添加策略信息到最终结果中
    result["strategy_used"] = strategy

    print(f"  [Detect Module] 策略 '{strategy}' 检测完成。")
    return result
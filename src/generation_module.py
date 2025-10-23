# src/generation_module.py
import json
import logging
from src.api_client import global_api_client
from src.prompts import GENERATION_SYSTEM_PROMPT, get_generation_user_prompt
from config import GENERATOR_MODEL_NAME, REQUEST_TEMPERATURE

logger = logging.getLogger(__name__)


def generate_answer(context: str, question: str) -> dict:
    """
    调用LLM（生成器），根据上下文回答问题。

    返回：
        一个包含答案和元数据(metadata)的字典。
    """
    print(f"正在生成答案...")

    if global_api_client is None:
        print("错误：API 客户端未初始化。")
        return {
            "status": "error",
            "answer": None,
            "error_message": "API Client not initialized."
        }

    # 1. 准备 Prompts
    system_prompt = GENERATION_SYSTEM_PROMPT
    user_prompt = get_generation_user_prompt(context, question)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 2. 调用 API (使用封装的客户端)
    try:
        response = global_api_client.create_chat_completion(
            model=GENERATOR_MODEL_NAME,
            messages=messages,
            temperature=REQUEST_TEMPERATURE
        )

        # 3. 解析响应
        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        model_used = response.model

        print(f"答案生成成功。")

        return {
            "status": "success",
            "answer": answer,
            "metadata": {
                "tokens_used": tokens_used,
                "model_used": model_used
            }
        }

    except Exception as e:
        print(f"答案生成失败: {e}")
        logger.error(f"调用DeepSeek API失败 (生成阶段): {e}", exc_info=True)
        return {
            "status": "error",
            "answer": None,
            "error_message": str(e)
        }
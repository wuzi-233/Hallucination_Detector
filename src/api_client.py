# src/api_client.py
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIError
import time
import sys
from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    REQUEST_TIMEOUT_SECONDS,
    MAX_RETRY_ATTEMPTS
)


class APIClient:
    """
    封装了DeepSeek API客户端的类。

    """
    def __init__(self, api_key, base_url, timeout):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.client = self._initialize_client()

    def _initialize_client(self):
        """
        内部方法：初始化 OpenAI 兼容的客户端。
        """
        if not self.api_key:
            print("API 密钥未提供。")
            return None

        print("正在初始化 DeepSeek API 客户端...")
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            return client
        except Exception as e:
            print(f"客户端初始化失败: {e}")
            return None

    def get_client(self):
        """
        获取已初始化的客户端实例。
        """
        return self.client

    def check_connection(self):
        """
        尝试调用 API 以验证密钥和连接是否正常。
        """
        if self.client is None:
            print("连接检查失败：客户端未初始化。")
            return False

        print("正在检查 API连接和密钥...")
        try:
            self.client.models.list()
            print("API 连接成功。")
            return True
        except APIError as e:
            print(f"API 密钥无效")
            return False
        except Exception as e:
            print(f"API 连接失败: {e}")
            return False

    def create_chat_completion(self, model, messages, temperature, response_format=None):
        """
        封装了带重试逻辑的 chat.completions.create 调用。
        """
        if self.client is None:
            print("无法创建 completion：客户端未初始化。")
            raise ValueError("API Client not initialized")

        retries = MAX_RETRY_ATTEMPTS
        delay = 1  # 初始延迟1秒

        for attempt in range(retries):
            try:
                # 构造请求参数
                request_args = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature
                }
                if response_format:
                    request_args["response_format"] = response_format

                # 发起请求
                response = self.client.chat.completions.create(**request_args)

                return response  # 成功则返回

            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                print(f"API 调用失败 (尝试 {attempt + 1}/{retries}): {type(e).__name__}")
                if attempt == retries - 1:
                    print("API 调用达到最大重试次数，放弃。")
                    raise e  # 抛出最终错误

                print(f"将在 {delay} 秒后重试...")
                time.sleep(delay)
                delay *= 2  # 指数退避

            except APIError as e:
                print(f"DeepSeek API 内部错误: {e}")
                raise e  # API内部错误，重试可能无效

            except Exception as e:
                print(f"调用 API 时发生未知错误: {e}")
                raise e  # 其他未知错误


# --- 创建全局实例 ---
if DEEPSEEK_API_KEY:
    global_api_client = APIClient(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS
    )
else:
    print("未创建全局客户端，因为 API 密钥缺失。")
    global_api_client = None
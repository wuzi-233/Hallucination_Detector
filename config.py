# config.py

# --- 1. 加载环境变量 ---
# print("配置模块(config.py)已加载...")

# --- 2. API 核心配置 ---
DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# --- 3. 模型配置 ---
GENERATOR_MODEL_NAME = "deepseek-chat"
DETECTOR_MODEL_NAME = "deepseek-chat"

# --- 4. API 请求参数 ---
REQUEST_TEMPERATURE = 0.0
REQUEST_TEMPERATURE_CREATIVE = 0.7
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRY_ATTEMPTS = 3
DEFAULT_MAX_TOKENS = 1024

# --- 5. 检测策略枚举 ---
class DetectionStrategy:
    DIRECT_ASK = "direct_ask"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"

DEFAULT_DETECTION_STRATEGY = DetectionStrategy.CHAIN_OF_THOUGHT

print(f"默认生成模型: {GENERATOR_MODEL_NAME}")
print(f"默认评估模型: {DETECTOR_MODEL_NAME}")
print(f"默认检测策略: {DEFAULT_DETECTION_STRATEGY}")
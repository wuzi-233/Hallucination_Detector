# main.py
import json
import sys
from config import DetectionStrategy, DEEPSEEK_API_KEY  # DEEPSEEK_API_KEY 会从 config.py 导入
from src.generation_module import generate_answer
from src.detection_module import detect_hallucination

try:
    from src.api_client import global_api_client

    if global_api_client is None:
        raise ImportError("API Client is None due to missing API Key")
except ImportError:
    print("错误：API 客户端未能初始化。")
    sys.exit(1)  # 退出程序

# --- 测试数据集 ---
TEST_CASES = [
    {
        "id": "case_001 (忠实)",
        "context": (
            "马里亚纳海沟是地球上已知最深的海沟，位于西太平洋，"
            "其最深处被称为“挑战者深渊”，深度约为11,034米。"
            "这个深度超过了珠穆朗玛峰的海拔高度。"
        ),
        "question": "马里亚纳海沟的最深处叫什么？有多深？"
    },
    {
        "id": "case_002 (幻觉)",
        "context": (
            "太阳系有八大行星，按离太阳的距离从近到远，"
            "它们依次为水星、金星、地球、火星。"
            "火星之外是木星和土星。"
        ),
        "question": "火星的平均温度是多少？"  # 上下文未提供
    },
    {
        "id": "case_003 (细微幻觉)",
        "context": (
            "AIGC（生成式人工智能）是一种能够创建新内容（如文本、图像或音乐）的AI。"
            "著名的模型包括OpenAI的GPT系列和Google的Gemini。"
        ),
        "question": "AIGC模型有哪些？"
        # 模型可能会回答 "GPT、Gemini 和 Claude" (Claude是幻觉)
    }
]

# --- 要运行的检测策略列表 ---
STRATEGIES_TO_RUN = [
    DetectionStrategy.DIRECT_ASK,
    DetectionStrategy.CHAIN_OF_THOUGHT,
    DetectionStrategy.FEW_SHOT
]


def run_full_pipeline(test_case: dict):
    """
    对单个测试用例执行“生成”和“多策略检测”的完整流水线。
    """
    case_id = test_case["id"]
    context = test_case["context"]
    question = test_case["question"]

    print("\n" + "=" * 20 + f" [正在处理测试案例: {case_id}] " + "=" * 20)
    print(f"[上下文]: {context[:100]}...")
    print(f"[问题]: {question}")

    # --- 阶段一：生成答案 ---
    print("\n[阶段 1] 正在调用生成器模型...")
    generation_result = generate_answer(context, question)

    if generation_result["status"] == "error":
        print(f"[阶段 1] 答案生成失败: {generation_result['error_message']}")
        return

    generated_answer = generation_result["answer"]
    print(f"[阶段 1] 生成的答案: \n{generated_answer}")

    # --- 阶段二：使用所有策略进行检测 ---
    print("\n[阶段 2] 正在调用检测器模型...")

    all_detection_results = []

    for strategy in STRATEGIES_TO_RUN:
        try:
            detection_result = detect_hallucination(context, generated_answer, strategy)
            all_detection_results.append(detection_result)
        except Exception as e:
            print(f"  [Main] 运行策略 {strategy} 时出错: {e}")
            all_detection_results.append({
                "strategy_used": strategy,
                "is_hallucination": "error",
                "explanation": f"执行时发生致命错误: {e}"
            })

    # --- 阶段三：打印评估报告 ---
    print("\n" + "-" * 15 + f" [案例 {case_id} 评估报告] " + "-" * 15)
    for res in all_detection_results:
        print(f"\n策略: {res.get('strategy_used')}")
        print(f"  是否幻觉: {res.get('is_hallucination')}")
        print(f"  解释: {res.get('explanation')}")
        if "thought_process" in res:
            print(f"  思考过程: {res.get('thought_process')[:150]}...")

    print("=" * (42 + len(case_id)))


def main():
    print("=" * 60)
    print("大语言模型幻觉检测系统")
    print("=" * 60)

    # 启动时检查连接
    print("正在检查API连接...")
    if not global_api_client.check_connection():
        print("API 连接失败或密钥无效，程序终止。")
        sys.exit(1)

    # 遍历所有测试用例
    for case in TEST_CASES:
        run_full_pipeline(case)

    print("\n" + "=" * 60)
    print("所有测试案例处理完毕。")
    print("=" * 60)


if __name__ == "__main__":
    main()
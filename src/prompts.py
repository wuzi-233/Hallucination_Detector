# src/prompts.py

"""
模块一：答案生成 (Generation)
"""

GENERATION_SYSTEM_PROMPT = (
    "你是一个严谨的、遵循指令的问答助手。"
    "你的任务是*严格*、*唯一*且*完全*根据用户提供的[上下文]来回答[问题]。"
    "在你的回答中，绝对不允许包含任何[上下文]之外的知识或信息。"
    "如果[上下文]中没有足够的信息来回答[问题]，你必须明确地、只回答：'根据上下文，我无法回答该问题'。"
    "不要添加任何额外的礼貌用语或解释，除非上下文中有相关信息。"
)


def get_generation_user_prompt(context: str, question: str) -> str:
    """
    生成“答案生成”阶段的用户提示词
    """
    return f"""
    [上下文]:
    {context}

    ---
    [问题]:
    {question}
    """


"""
模块二：幻觉检测 (Detection)

"""

# --- 策略 A: 直接询问 ---

DETECTION_DIRECT_ASK_SYSTEM_PROMPT = (
    "你是一个严谨的评估助手。"
    "你的任务是判断一个[答案]是否*完全*由[上下文]支持。"
    "你必须以严格的JSON格式输出你的评估结果。"
    "评估标准：\n"
    "1. 忠实 (Faithful): [答案]中的所有信息都*严格*基于[上下文]推导或复述。\n"
    "2. 幻觉 (Hallucination): [答案]中包含任何[上下文]中*未提及*或*相矛盾*的信息。\n"
)


def get_detection_direct_ask_user_prompt(context: str, answer: str) -> str:
    return f"""
    [上下文]:
    {context}

    [答案]:
    {answer}

    ---
    请根据上述评估标准，判断[答案]是否为[幻觉]。
    请按照以下JSON格式返回你的评估结果 (is_hallucination 必须为 true 或 false)：
    {{
      "is_hallucination": <true_or_false>,
      "explanation": "<你的详细评估理由>"
    }}
    """


# --- 策略 B: 逐步思考 ---

DETECTION_COT_SYSTEM_PROMPT = (
    "你是一个极其严谨和细致的分析师。"
    "你的任务是评估一个[答案]是否忠实于一个[上下文]。"
    "你必须使用“逐步思考”的方法来分析，最后以JSON格式输出你的结论。"
)


def get_detection_cot_user_prompt(context: str, answer: str) -> str:
    return f"""
    [上下文]:
    {context}

    [答案]:
    {answer}

    ---
    请遵循以下步骤进行评估：

    步骤 1. [分析答案]: 将[答案]分解为核心信息点或断言。
    步骤 2. [逐条核对]: 检查[上下文]中是否有支持每个信息点或断言的明确证据。
    步骤 3. [形成结论]: 总结你的发现。如果所有信息点都得到了支持，则答案是'忠实'的。如果任何一个信息点在上下文中未提及或相矛盾，则答案包含'幻觉'。
    步骤 4. [输出JSON]: 仅输出一个JSON对象，包含你的最终评估。

    请严格按照以下JSON格式返回你的评估结果：
    {{
      "thought_process": "<这里是你详细的步骤1、2、3的分析过程，用于展示你的思考>",
      "is_hallucination": <true_or_false>,
      "explanation": "<这里是你的最终结论的简明解释>"
    }}
    """


# --- 策略 C: 少量样本 ---
# (这个Prompt非常长，因为它包含示例)

DETECTION_FEW_SHOT_SYSTEM_PROMPT = (
    "你是一个严谨的评估助手，你通过学习示例来工作。"
    "你的任务是判断一个[答案]是否*完全*由[上下文]支持。"
    "你必须以严格的JSON格式输出你的评估结果。"
)


def get_detection_few_shot_user_prompt(context: str, answer: str) -> str:
    # 示例被硬编码在Prompt中，这是一种常见的Few-Shot方法
    return f"""
    以下是一些评估示例：

    --- 示例 1 (忠实) ---
    [上下文]: 
    "苹果公司在2007年发布了第一代iPhone，由史蒂夫·乔布斯揭晓。"
    [答案]: 
    "第一代iPhone是在2007年由苹果公司发布的。"
    [评估JSON]:
    {{
      "is_hallucination": false,
      "explanation": "答案中的所有信息（'iPhone', '2007年', '苹果公司'）在上下文中都得到了支持。"
    }}

    --- 示例 2 (幻觉) ---
    [上下文]:
    "长城是中国的古代防御工程，横跨数千公里。"
    [答案]:
    "长城是中国的防御工程，由秦始皇建造，目的是抵御来自北方的蒙古人。"
    [评估JSON]:
    {{
      "is_hallucination": true,
      "explanation": "答案中提到的'由秦始皇建造'和'抵御蒙古人'在[上下文]中并未提及。这是幻觉。"
    }}

    --- 示例 3 (幻觉) ---
    [上下文]:
    "咖啡因是一种中枢神经兴奋剂。它天然存在于咖啡豆和茶叶中。"
    [答案]:
    "咖啡因是一种兴奋剂，存在于咖啡豆、茶叶和可可豆中。"
    [评估JSON]:
    {{
      "is_hallucination": true,
      "explanation": "答案中提到的'可可豆'在[上下文]中并未提及。"
    }}

    --- 示例 4 (忠实 - 无法回答) ---
    [上下文]:
    "亚马逊河是世界上流域面积最广的河流。"
    [答案]:
    "根据上下文，我无法回答该问题"
    [评估JSON]:
    {{
      "is_hallucination": false,
      "explanation": "模型正确地指出无法回答。它没有捏造信息。"
    }}

    --- 结束示例 ---

    现在，请评估以下[任务]：

    [任务]:

    [上下文]:
    {context}

    [答案]:
    {answer}

    ---
    请按照上述示例的JSON格式，对这个[任务]进行评估：
    [评估JSON]:
    """
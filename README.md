大语言模型幻觉检测器 (LLM Hallucination Detector)

本项目是一个用于评估和检测大语言模型（LLM）在“上下文相关问答”（Contextual Q&A）任务中是否产生“幻觉”的系统。

使用“生成器-检测器” (Generator-Detector) 框架，通过一个两阶段的流水线来评估一个LLM（作为生成器）生成的答案是否完全忠实于所提供的上下文。

核心特性：

两阶段评估流水线：分离“答案生成”与“幻觉检测”两个阶段。

多策略检测：内置三种不同的检测策略（DIRECT_ASK, CHAIN_OF_THOUGHT, FEW_SHOT）对答案进行交叉验证，以提高评估的准确性。

提示工程：系统提示词（System Prompts），用于严格约束生成器LLM，并精确指导检测器LLM进行分析。

API客户端：封装了与DeepSeek API的交互，包含自动重试、超时处理和连接验证。

JSON解析：自动清理和解析LLM返回的不规范JSON响应。

项目结构:
```
Hallucination_Detector/
│
├── main.py                 # 项目主入口，定义测试用例和执行流水线
├── config.py               # 配置文件（API密钥、模型名称、温度等）
├──.gitignore               # 忽略Git提交的文件
│
└── src/
    ├── api_client.py         # 封装了DeepSeek API的客户端
    ├── generation_module.py  # 阶段一：答案生成模块
    ├── detection_module.py   # 阶段二：幻觉检测模块
    ├── prompts.py            # 定义所有与LLM交互的System Prompts
    └── __pycache__/
```

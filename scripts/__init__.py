"""命令行入口脚本包。

提供 ``python -m scripts.<name>`` 形式的命令行工具：

- ``build_index``：从 ``data/chunks.jsonl`` 构建本地向量索引；
- ``ask``：命令行问答入口，打印回答与引用摘要（--json 输出 AnswerResult）；
- ``serve``：Web 界面与 JSON API（与成员 D 协同）；
- ``eval_answer``：跑 ``eval/questions.jsonl`` 并输出基础指标。

成员 E 负责 build_index / ask / eval_answer 的维护与「快速开始」可复现。
"""


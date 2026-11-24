"""命令行入口脚本包。

提供 ``python -m scripts.<name>`` 形式的命令行工具：

- ``build_index``：从 ``data/chunks.jsonl`` 构建本地向量索引；
- ``build_life_index``：重建校园生活库 BM25 索引；
- ``crawl_wall``：从校园墙提炼问题集（脱敏 + 去重）；
- ``ask``：命令行问答入口，打印回答与引用摘要（--json / --smart）；
- ``serve``：Web 界面与 JSON API；
- ``eval_answer``：跑 ``eval/questions.jsonl`` 并输出基础指标。

成员 E 负责 build_index / ask / eval_answer 的维护与「快速开始」可复现。
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows + 中文路径下，editable 安装的 .pth 可能不被 site 处理；
# 在包初始化时兜底把 src 加进 sys.path，保证 `python -m scripts.*` 都能 import。
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


"""Pytest 路径引导。

`pyproject.toml` 只把 `src/` 加入 pythonpath，而 `scripts/` 包位于仓库
根目录。这里把仓库根目录加入 `sys.path`，使 `from scripts import serve`
等导入在测试中可用。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

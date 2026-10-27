"""基础配置：路径、模型名称、检索参数。

尽量保持简单，集中管理少量关键参数，方便后续在
service / scripts 中复用，而不是搞一整套复杂配置系统。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# 加载 .env，允许在本地直接配置密钥与模型名
load_dotenv()


# 仓库根目录与数据目录约定
# config.py 位于 <repo>/src/campus_qa/config.py，因此：
#   parents[0] = src/campus_qa，parents[1] = src，parents[2] = <repo>
SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"


# 兼容常见写法：只写模型短名时自动补全 HuggingFace 组织名。
_EMBEDDING_MODEL_ALIASES = {
    "bge-small-zh-v1.5": "BAAI/bge-small-zh-v1.5",
    "bge-base-zh-v1.5": "BAAI/bge-base-zh-v1.5",
    "bge-large-zh-v1.5": "BAAI/bge-large-zh-v1.5",
}


def _env(*names: str, default: str = "") -> str:
    """按顺序返回第一个非空环境变量，兼容多种命名习惯。"""

    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _resolve_llm() -> tuple[str, str, str]:
    """解析 LLM 的 (api_key, base_url, model)。

    优先级：显式 LLM_* → DEEPSEEK_* → OPENAI_*。
    这样既符合文档约定的变量名，也能直接复用现有 .env。
    """

    explicit_key = _env("LLM_API_KEY")
    if explicit_key:
        return (
            explicit_key,
            _env("LLM_BASE_URL", "OPENAI_BASE_URL"),
            _env("CHAT_MODEL", "LLM_MODEL", "OPENAI_MODEL", default="gpt-4.1-mini"),
        )

    deepseek_key = _env("DEEPSEEK_API_KEY")
    if deepseek_key:
        return (
            deepseek_key,
            _env("LLM_BASE_URL", "DEEPSEEK_BASE_URL"),
            _env("CHAT_MODEL", "DEEPSEEK_MODEL", default="deepseek-chat"),
        )

    return (
        _env("OPENAI_API_KEY"),
        _env("LLM_BASE_URL", "OPENAI_BASE_URL"),
        _env("CHAT_MODEL", "OPENAI_MODEL", default="gpt-4.1-mini"),
    )


def _resolve_embedding_model() -> str:
    """解析向量模型名称，并为常见短名补全组织前缀。"""

    raw = _env("EMBEDDING_MODEL", "EMBED_MODEL", default="BAAI/bge-small-zh-v1.5")
    return _EMBEDDING_MODEL_ALIASES.get(raw, raw)


@dataclass(frozen=True)
class Settings:
    """项目运行所需的基础配置。

    不做复杂层级：只包含 LLM、检索参数和数据路径。
    """

    llm_api_key: str
    chat_model: str = "gpt-4.1-mini"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_base_url: Optional[str] = None

    # 检索相关参数
    retrieval_top_k: int = 5
    retrieval_per_doc: int = 2
    retrieval_score_threshold: float = 0.35

    # 数据路径约定
    data_dir: Path = DATA_DIR
    chunks_path: Path = DATA_DIR / "chunks.jsonl"
    index_dir: Path = DATA_DIR / "index"


def load_settings() -> Settings:
    """从环境变量构建 `Settings`，提供默认值以便本地快速跑通。

    环境变量约定：
    - LLM_API_KEY / LLM_BASE_URL / CHAT_MODEL
      （兼容 DEEPSEEK_API_KEY、OPENAI_API_KEY 等既有配置）
    - EMBEDDING_MODEL（兼容 EMBED_MODEL）
    - RETRIEVAL_TOP_K
    - RETRIEVAL_PER_DOC
    - RETRIEVAL_SCORE_THRESHOLD
    """

    api_key, base_url, chat_model = _resolve_llm()

    return Settings(
        llm_api_key=api_key,
        chat_model=chat_model,
        embedding_model=_resolve_embedding_model(),
        llm_base_url=base_url or None,
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
        retrieval_per_doc=int(os.getenv("RETRIEVAL_PER_DOC", "2")),
        retrieval_score_threshold=float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.35")),
    )


# 直接暴露一个全局设置实例，便于在其他模块中引用。
settings = load_settings()

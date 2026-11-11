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
    "bge-m3": "BAAI/bge-m3",
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


def _env_bool(name: str, default: bool) -> bool:
    """解析布尔型环境变量，兼容 1/0、true/false、yes/no、on/off。"""

    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    # 多路召回实验最优配置（评测对比见 .evallogs/scan_*）：
    # per_doc=4 / bm25_top_k=10 / rrf_k=60 → Recall@5 0.862、grounded 0.836。
    retrieval_top_k: int = 5
    retrieval_per_doc: int = 4
    retrieval_score_threshold: float = 0.35
    retrieval_enable_bm25: bool = True
    retrieval_bm25_top_k: int = 10
    retrieval_rrf_k: float = 60.0
    retrieval_dedup_threshold: float = 0.85
    retrieval_enable_rewrite: bool = True
    retrieval_enable_llm_rewrite: bool = False
    # 领域相关性判定阈值：非制度主题且 top-1 低于该值则拒答（no_evidence_in_corpus）。
    domain_score_threshold: float = 0.68

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
    - RETRIEVAL_ENABLE_BM25（可选，多路召回开关，默认 true）
    - RETRIEVAL_BM25_TOP_K（可选，BM25 路候选数，默认 10）
    - RETRIEVAL_RRF_K（可选，RRF 融合常数，默认 60）
    - RETRIEVAL_DEDUP_THRESHOLD（可选，跨文档同主题去重重合率阈值，默认 0.85）
    - RETRIEVAL_ENABLE_REWRITE（可选，查询规则改写开关，默认 true）
    - RETRIEVAL_ENABLE_LLM_REWRITE（可选，口语 query 的 LLM 改写开关，默认 false）
    - RETRIEVAL_INDEX_DIR（可选，向量索引目录，默认 data/index）
    """

    api_key, base_url, chat_model = _resolve_llm()

    index_dir = Path(os.getenv("RETRIEVAL_INDEX_DIR", str(DATA_DIR / "index")))

    return Settings(
        llm_api_key=api_key,
        chat_model=chat_model,
        embedding_model=_resolve_embedding_model(),
        llm_base_url=base_url or None,
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
        retrieval_per_doc=int(os.getenv("RETRIEVAL_PER_DOC", "4")),
        retrieval_score_threshold=float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.35")),
        retrieval_enable_bm25=_env_bool("RETRIEVAL_ENABLE_BM25", default=True),
        retrieval_bm25_top_k=int(os.getenv("RETRIEVAL_BM25_TOP_K", "10")),
        retrieval_rrf_k=float(os.getenv("RETRIEVAL_RRF_K", "60.0")),
        retrieval_dedup_threshold=float(os.getenv("RETRIEVAL_DEDUP_THRESHOLD", "0.85")),
        retrieval_enable_rewrite=_env_bool("RETRIEVAL_ENABLE_REWRITE", default=True),
        retrieval_enable_llm_rewrite=_env_bool("RETRIEVAL_ENABLE_LLM_REWRITE", default=False),
        domain_score_threshold=float(os.getenv("DOMAIN_SCORE_THRESHOLD", "0.68")),
        index_dir=index_dir,
    )


# 直接暴露一个全局设置实例，便于在其他模块中引用。
settings = load_settings()

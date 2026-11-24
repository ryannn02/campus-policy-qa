"""脱敏与隐私边界。

两条硬约束（对应 M5 需求）：

1. **入库前脱敏**：校园墙原始内容在进入问题集之前，手机号、学号/工号、
   QQ/微信号、身份证号、姓名全部替换为占位符，原文不落盘。
2. **渠道侧不查人**：只回答「问题」，不提供任何查询他人信息的能力；
   命中「人肉 / 开盒 / 查某人手机号」这类请求时直接拒绝。

脱敏是纯规则实现，不依赖模型，便于在导入脚本、QQ 渠道、日志三处复用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---- 脱敏规则：顺序敏感（身份证 18 位要先于手机号 11 位）----
_REDACTION_RULES: List[Tuple[str, str, str]] = [
    # 身份证号：17 位数字 + 校验位（数字或 X）
    ("id_card", r"(?<![0-9A-Za-z])\d{17}[\dXx](?![0-9A-Za-z])", "[身份证号]"),
    # 手机号：1 开头 11 位，前后不能紧邻数字
    ("phone", r"(?<!\d)1[3-9]\d{9}(?!\d)", "[手机号]"),
    # 学号 / 工号：必须带前缀，避免误伤普通数字
    ("student_id", r"(学号|工号)\s*[:：]?\s*([A-Za-z0-9]{6,14})", r"\1[学号]"),
    # QQ 号
    ("qq", r"(?i)(qq|扣扣|企鹅)\s*(号|账号)?\s*[:：]?\s*(\d{5,11})", r"\1[QQ号]"),
    # 微信号 / 微信名
    (
        "wechat",
        r"(?i)(微信|wechat|weixin|vx)\s*(号|账号)?\s*[:：]?\s*([A-Za-z][A-Za-z0-9_-]{4,19})",
        r"\1[微信号]",
    ),
    # 自报姓名：我叫张三 / 姓名：张三 / 我是张三（后面必须跟标点或结尾，
    # 避免把「我是大二的学生」里的「大二」当成姓名）
    (
        "name",
        r"(我叫|我是|姓名\s*[:：]?|名字\s*[:：]?)\s*([\u4e00-\u9fa5]{2,4})(?=[，。！？、,.!?\s]|$)",
        r"\1[姓名]",
    ),
    # 称呼：张同学 / 李老师 → 某同学 / 某老师
    ("name", r"([\u4e00-\u9fa5])(同学|老师|师傅|学长|学姐|阿姨)", r"某\2"),
]

_COMPILED_RULES = [(name, re.compile(pattern), repl) for name, pattern, repl in _REDACTION_RULES]


# ---- 隐私请求拦截：只拦「查别人」，不拦「查公开办公电话」----
_PRIVACY_PATTERNS: List[Tuple[str, str]] = [
    ("doxxing", r"人肉|开盒|社工库|查户籍|查档案"),
    (
        "someone_contact",
        r"(别人|他人|某人|同学|室友|他|她|这个人|那个人|此人)"
        r"[^。？！?]{0,8}(手机号|电话号|微信号|qq号|身份证|家庭住址|宿舍号|父母电话)",
    ),
    (
        "someone_id",
        r"(别人|他人|某人|同学|室友|这个人|那个人|此人)[^。？！?]{0,6}(学号|身份证号|住址)",
    ),
    ("locate", r"(定位|追踪)[^。？！?]{0,6}(他|她|某人|同学|这个人)"),
    # 「查/要/给我」+ 手机号/身份证/住址：明显是索取他人隐私；
    # 不拦「我的学号怎么查」这类自助查询，所以不含「学号」。
    ("request_lookup", r"(查|要|给我|发我)[^。？！?]{0,6}(手机号|身份证号|家庭住址)"),
    # 「把张三的手机号发我」这类祈使句
    ("ask_others", r"(把|将)[^。？！?]{0,8}(手机号|身份证号|家庭住址)"),
]

_PRIVACY_COMPILED = [(name, re.compile(pattern, re.I)) for name, pattern in _PRIVACY_PATTERNS]


@dataclass
class RedactionResult:
    """脱敏结果：文本 + 各类命中次数（用于导入报表，不含原文）。"""

    text: str
    hits: Dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.hits)

    @property
    def total(self) -> int:
        return sum(self.hits.values())


def redact(text: str) -> RedactionResult:
    """对单条文本做脱敏，返回替换后的文本与命中统计。"""

    if not text:
        return RedactionResult(text=text or "")

    hits: Dict[str, int] = {}
    result = text
    for name, pattern, repl in _COMPILED_RULES:
        result, count = pattern.subn(repl, result)
        if count:
            hits[name] = hits.get(name, 0) + count

    # 掩码后可能出现连续的占位符噪音，收敛多余空白但不改语义
    result = re.sub(r"[ \t]{2,}", " ", result).strip()
    return RedactionResult(text=result, hits=hits)


def redact_text(text: str) -> str:
    """只要脱敏后的文本，便于链式调用。"""

    return redact(text).text


def privacy_request_reason(text: str) -> str | None:
    """命中「查询他人隐私」时返回规则名，否则 None。"""

    if not text:
        return None
    for name, pattern in _PRIVACY_COMPILED:
        if pattern.search(text):
            return name
    return None


def is_privacy_request(text: str) -> bool:
    return privacy_request_reason(text) is not None


def scrub_mapping(record: dict, fields: List[str]) -> Tuple[dict, Dict[str, int]]:
    """批量脱敏字典中的指定字段，返回新字典与累计命中统计。"""

    cleaned = dict(record)
    total_hits: Dict[str, int] = {}
    for field_name in fields:
        value = cleaned.get(field_name)
        if not isinstance(value, str):
            continue
        result = redact(value)
        cleaned[field_name] = result.text
        for key, count in result.hits.items():
            total_hits[key] = total_hits.get(key, 0) + count
    return cleaned, total_hits

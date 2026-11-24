"""意图路由：制度 / 生活 / 工具 / 闲聊。

设计原则：

- **制度优先**。识别不出来时兜底 `POLICY`，把问题交给原来的 service，
  这样拒答与召回行为与 M3 完全一致，不会因为新增路由引入新的误答。
- **工具优先于生活**。「图书馆几点开门」是生活，「现在几点」是工具，
  所以时间类工具只在出现明确表述（现在几点 / 今天几号）时才触发。
- **多轮可继承**。「那快递呢？」本身没有关键词，靠上一轮意图接住。

路由是纯规则实现，零模型依赖，可单测、可解释（命中词会一起返回）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .contracts import Intent


# ---- 工具识别：词 → 工具名 ----
_TOOL_PATTERNS: List[Tuple[str, str]] = [
    ("weather", r"天气|气温|温度|下雨|降雨|带伞|冷不冷|热不热|风力|几度"),
    ("campus_time", r"现在几点|几点了|当前时间|今天几号|今天日期|星期几|周几"),
    ("campus_calendar", r"第几周|校历|学期|开学|放假|假期|什么时候开学"),
    ("campus_phones", r"电话|联系方式|号码|找谁|打给谁|值班"),
]

_TOOL_COMPILED = [(name, re.compile(pattern)) for name, pattern in _TOOL_PATTERNS]


# ---- 关键词表：(正则, 权重) ----
_POLICY_PATTERNS: List[Tuple[str, int]] = [
    (r"制度|规定|条例|办法|细则|章程|守则|文件", 3),
    (r"学籍|学分|选课|课程|培养方案|教学计划", 2),
    (r"补考|重修|缓考|考试|成绩|绩点|挂科", 2),
    (r"休学|复学|退学|转专业|转学|延期毕业", 2),
    (r"奖学金|助学金|助学贷款|评优|评先|困难补助", 2),
    (r"处分|违纪|通报批评|申诉|复议", 2),
    (r"毕业论文|学位|答辩|毕业|结业", 2),
    (r"实习|实践|请假|考勤|旷课|销假", 2),
    (r"学费|收费|退费|报销|经费", 2),
    (r"采购|招标|投标|合同|预算|资产|设备|实验室安全", 2),
    (r"基建|基本建设|维修项目|维修工程|零星维修|归口管理|立项|施工单位|工程变更|竣工", 2),
    (r"档案|户口|政审|党组织|团组织|入党|入团", 2),
    (r"借阅|借书|还书|逾期|赔偿|遗失", 2),
    (r"宿舍管理|住宿|寝室管理|调宿|退宿|寝室长|宿舍长|楼层长", 2),
    (r"保研|推免|考研|就业|三方协议|报到证", 2),
]

_LIFE_PATTERNS: List[Tuple[str, int]] = [
    (r"食堂|餐厅|吃饭|早餐|午餐|晚餐|夜宵|饭菜|打饭", 3),
    (r"快递|驿站|取件|寄件|快递柜|外卖", 3),
    (r"校园卡|一卡通|充值|挂失|补卡|余额|刷卡", 3),
    (r"热水|洗澡|浴室|开水|饮水机|直饮水", 2),
    (r"校园网|wifi|无线网|宽带|上网|流量", 2),
    (r"打印|复印|扫描|文印", 2),
    (r"超市|便利店|小卖部|购物", 2),
    (r"医务室|校医院|看病|开药|药店|医保", 2),
    (r"洗衣|干洗|洗衣房", 2),
    (r"空调|报修|维修|水电|停水|停电", 2),
    (r"宿舍|床位|寝室", 1),
    (r"班车|校车|停车|充电桩|电动车", 2),
    (r"理发|健身|游泳|操场|球场|体育馆|体育场馆|体育场|篮球场|足球场|健身房|乒乓球", 2),
    (r"自习|图书馆开放|开放时间|闭馆|开门|关门|座位|占座", 3),
    (r"垃圾分类|垃圾投放|垃圾桶|分类投放", 2),
]

_CHITCHAT_PATTERNS: List[Tuple[str, int]] = [
    (r"^你好|^您好|^hi|^hello|^嗨|^在吗", 2),
    (r"谢谢|感谢|多谢|辛苦", 2),
    (r"再见|拜拜|晚安", 2),
    (r"你是谁|你叫什么|你能做什么|你会什么|你是什么", 3),
    (r"早上好|晚上好|下午好", 2),
]

_POLICY_COMPILED = [(re.compile(p), w) for p, w in _POLICY_PATTERNS]
_LIFE_COMPILED = [(re.compile(p), w) for p, w in _LIFE_PATTERNS]
_CHITCHAT_COMPILED = [(re.compile(p), w) for p, w in _CHITCHAT_PATTERNS]

# 明确的「接着上一轮问」信号
_FOLLOWUP_PREFIX = re.compile(r"^(那|那么|还有|再问|另外|它|这个|那个|这种|那如果)")
_FOLLOWUP_TAIL = re.compile(r"呢[？?]?$")


@dataclass(frozen=True)
class RouteDecision:
    """路由结果。`matched` 保留命中词，便于日志与评测归因。"""

    intent: Intent
    confidence: float
    matched: List[str] = field(default_factory=list)
    reason: str = "keyword"
    tool: Optional[str] = None


def _match_tool(text: str) -> Optional[str]:
    for name, pattern in _TOOL_COMPILED:
        if pattern.search(text):
            return name
    return None


def _score(patterns: List[Tuple[re.Pattern[str], int]], text: str) -> Tuple[int, List[str]]:
    total = 0
    matched: List[str] = []
    for pattern, weight in patterns:
        found = pattern.search(text)
        if found:
            total += weight
            matched.append(found.group(0))
    return total, matched


def is_followup(question: str, last_question: Optional[str] = None) -> bool:
    """判断是否是对上一轮的追问（不依赖模型）。"""

    text = (question or "").strip()
    if not text:
        return False
    if _FOLLOWUP_PREFIX.search(text) or _FOLLOWUP_TAIL.search(text):
        return True
    # 极短的补充（「几点」「在哪」「多少钱」）默认视为追问
    if last_question and len(text) <= 6:
        return True
    return False


def classify(
    question: str,
    previous_intent: Optional[Intent] = None,
    last_question: Optional[str] = None,
) -> RouteDecision:
    """对问题做意图分类。

    `previous_intent` 只用于「刚才问过同类问题、这轮没有关键词」的继承，
    不会覆盖任何有明确关键词的判断。
    """

    text = (question or "").strip()
    if not text:
        return RouteDecision(intent=Intent.POLICY, confidence=0.0, reason="empty")

    tool = _match_tool(text)

    policy_score, policy_hits = _score(_POLICY_COMPILED, text)
    life_score, life_hits = _score(_LIFE_COMPILED, text)
    chat_score, chat_hits = _score(_CHITCHAT_COMPILED, text)

    # 1) 工具：只有明确时间/天气/校历/电话表述才抢路由
    if tool:
        return RouteDecision(
            intent=Intent.TOOL,
            confidence=min(1.0, 0.5 + 0.1 * len(text) / 10),
            matched=[tool],
            reason="tool",
            tool=tool,
        )

    # 2) 制度（含「规定/办法」等强词）
    if policy_score >= 3 or (policy_score > 0 and policy_score >= life_score):
        return RouteDecision(
            intent=Intent.POLICY,
            confidence=min(1.0, 0.4 + policy_score / 6),
            matched=policy_hits,
            reason="policy_keyword",
        )

    # 3) 生活
    if life_score > 0:
        return RouteDecision(
            intent=Intent.LIFE,
            confidence=min(1.0, 0.4 + life_score / 6),
            matched=life_hits,
            reason="life_keyword",
        )

    # 4) 闲聊
    if chat_score > 0:
        return RouteDecision(
            intent=Intent.CHITCHAT,
            confidence=min(1.0, 0.4 + chat_score / 4),
            matched=chat_hits,
            reason="chitchat",
        )

    # 5) 无关键词：追问继承上一轮意图
    if previous_intent is not None and is_followup(text, last_question):
        return RouteDecision(
            intent=previous_intent,
            confidence=0.5,
            matched=[],
            reason="inherit",
        )

    # 6) 兜底：交给制度链路（保持 M3 行为）
    return RouteDecision(intent=Intent.POLICY, confidence=0.3, matched=[], reason="fallback")

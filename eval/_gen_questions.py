"""生成评测集 questions.jsonl（成员 C，M2 任务 2）。

从 data/chunks.jsonl 的真实条款出发，按 chunk_id 精确标注 gold
（document_id / snippet / page / section）。可回答题 gold 覆盖 100%，
四类（answerable / multi-hop / unanswerable / ambiguous）齐全。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
OUT_PATH = ROOT / "eval" / "questions.jsonl"


def _coerce_page(value: object):
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        m = re.search(r"\d+", value)
        return int(m.group()) if m else None
    return None


def load_chunks() -> dict:
    chunks = {}
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            chunks[r.get("chunk_id")] = r
    return chunks


# 评测集定义：每项 (id, question, category, should_answer, [gold chunk_id...])
ITEMS = [
    # ---------- answerable：学籍 / 入学 / 注册 ----------
    ("A001", "新生因故不能按期入学怎么办？", "answerable", True, ["8beaca14b3b6-00011"]),
    ("A002", "新生入学资格如何审查？", "answerable", True, ["8beaca14b3b6-00012"]),
    ("A003", "新生可以申请保留入学资格吗？保留期间有什么规定？", "answerable", True, ["8beaca14b3b6-00013"]),
    ("A004", "新生入学后学校会进行哪些复查？", "answerable", True, ["8beaca14b3b6-00014"]),
    ("A005", "每学期开学如何办理注册手续？", "answerable", True, ["8beaca14b3b6-00015"]),
    ("A006", "家庭经济困难学生如何办理注册？", "answerable", True, ["8beaca14b3b6-00122"]),
    ("A007", "哪些原因可以申请保留入学资格？", "answerable", True, ["8beaca14b3b6-00070"]),
    # ---------- answerable：学分 / 辅修 / 绩点 ----------
    ("A008", "学生可以辅修其他专业或跨校修读课程吗？", "answerable", True, ["8beaca14b3b6-00018"]),
    ("A009", "创新创业、社会实践等成果可以折算学分吗？", "answerable", True, ["8beaca14b3b6-00019"]),
    ("A010", "平均学分绩点有什么用途？", "answerable", True, ["8beaca14b3b6-00084"]),
    ("A011", "学生思想品德如何考核？", "answerable", True, ["8beaca14b3b6-00017"]),
    ("A012", "符合什么条件可以申请课程免修？", "answerable", True, ["16f8710dd84e-00717"]),
    # ---------- answerable：学籍异动 ----------
    ("A013", "学生可以申请转专业吗？有什么限制？", "answerable", True, ["8beaca14b3b6-00023"]),
    ("A014", "符合什么条件可以申请转专业？", "answerable", True, ["8beaca14b3b6-00097"]),
    ("A015", "入伍学生转专业有什么特殊规定？", "answerable", True, ["8beaca14b3b6-00139"]),
    ("A016", "学生申请休学有什么规定？最长学习年限如何计算？", "answerable", True, ["8beaca14b3b6-00027"]),
    ("A017", "休学创业有什么特殊政策？", "answerable", True, ["8beaca14b3b6-00028"]),
    ("A018", "什么情况可以申请转学？", "answerable", True, ["8beaca14b3b6-00024"]),
    ("A019", "复学学生如何编入班级？", "answerable", True, ["8beaca14b3b6-00105"]),
    ("A020", "退役士兵复学有什么规定？", "answerable", True, ["8beaca14b3b6-00136"]),
    # ---------- answerable：毕业 / 结业 / 退学 ----------
    ("A021", "学生达到什么条件可以毕业？", "answerable", True, ["8beaca14b3b6-00033"]),
    ("A022", "什么情况会发结业证书？结业后可以补考吗？", "answerable", True, ["8beaca14b3b6-00034"]),
    ("A023", "学生退学后如何办理离校手续？", "answerable", True, ["8beaca14b3b6-00032"]),
    ("A024", "补考、重修获得的成绩如何记载？", "answerable", True, ["8beaca14b3b6-00020"]),
    ("A025", "学历证书、学位证书如何填写和颁发？", "answerable", True, ["8beaca14b3b6-00035"]),
    # ---------- answerable：课程考核 ----------
    ("A026", "课程考核有哪些方式？成绩如何评定？", "answerable", True, ["8beaca14b3b6-00076"]),
    ("A027", "什么情况可以申请缓考？", "answerable", True, ["8beaca14b3b6-00080"]),
    ("A028", "入伍学生如何办理缓考？", "answerable", True, ["8beaca14b3b6-00135"]),
    ("A029", "课程重修有次数限制吗？", "answerable", True, ["8beaca14b3b6-00095"]),
    ("A030", "考试违纪可以参加补考或重修吗？", "answerable", True, ["8beaca14b3b6-00233"]),
    ("A031", "补考课程时间冲突怎么办？", "answerable", True, ["f59c55255624-00900"]),
    ("A032", "考试作弊成绩如何处理？", "answerable", True, ["8beaca14b3b6-00081"]),
    ("A033", "哪些行为属于考试作弊？", "answerable", True, ["8beaca14b3b6-00229"]),
    ("A034", "考试结束后发现作弊怎么处理？", "answerable", True, ["8beaca14b3b6-00230"]),
    ("A035", "考试时手机等通讯设备如何处理？", "answerable", True, ["8beaca14b3b6-00199"]),
    ("A036", "对考试违规认定有异议如何申诉？", "answerable", True, ["8beaca14b3b6-00239"]),
    # ---------- answerable：纪律处分 / 申诉 ----------
    ("A037", "学校处分学生的原则是什么？", "answerable", True, ["8beaca14b3b6-00051"]),
    ("A038", "处分决定书包含哪些内容？", "answerable", True, ["8beaca14b3b6-00050"]),
    ("A039", "学生对处分有异议如何申诉？", "answerable", True, ["8beaca14b3b6-00059"]),
    ("A040", "学生无故缺席会有什么后果？", "answerable", True, ["8beaca14b3b6-00021"]),
    # ---------- answerable：奖励 / 奖学金 / 资助 ----------
    ("A041", "学校对优秀学生有哪些表彰奖励？", "answerable", True, ["8beaca14b3b6-00046"]),
    ("A042", "对学生的表彰奖励有哪些形式？", "answerable", True, ["8beaca14b3b6-00047"]),
    ("A043", "国家奖学金的申请条件是什么？", "answerable", True, ["48f3fcb23120-01146"]),
    ("A044", "优秀学生奖学金什么时候申报？", "answerable", True, ["68d3951a7876-01268"]),
    ("A045", "学校勤工助学管理的依据是什么？", "answerable", True, ["48db9eefd40f-01174"]),
    ("A046", "哪些学生属于家庭经济困难学生？", "answerable", True, ["0a91830dad84-02746"]),
    ("A047", "哪些情况不予临时困难补助？", "answerable", True, ["299fb57d063f-01245"]),
    # ---------- answerable：图书馆 ----------
    ("A048", "哪些人可以借书？", "answerable", True, ["76af297ad0fe-03660"]),
    ("A049", "用他人校园卡借书怎么处理？", "answerable", True, ["0a91830dad84-02799"]),
    ("A050", "把校园卡借给他人用会有什么后果？", "answerable", True, ["9bbe8160302a-03646"]),
    ("A051", "图书馆对文明借阅有什么要求？", "answerable", True, ["0a91830dad84-02789"]),
    # ---------- answerable：宿舍 / 生活园区 ----------
    ("A052", "宿舍区消防安全管理有什么规定？", "answerable", True, ["ea668a3c48a6-01001"]),
    ("A053", "宿舍内严禁哪些行为？", "answerable", True, ["ea668a3c48a6-01004"]),
    ("A054", "宿舍消防安全由哪些部门负责？", "answerable", True, ["ea668a3c48a6-01002"]),
    ("A055", "如何申请调整寝室床位？", "answerable", True, ["b0951330395c-01205"]),
    ("A056", "寝室长的主要职责是什么？", "answerable", True, ["b0951330395c-01207"]),
    ("A057", "寝室贵重物品带出需要什么手续？", "answerable", True, ["651f5d582cef-01428"]),
    # ---------- answerable：校园交通 ----------
    ("A058", "校内骑电动车要遵守什么规范？", "answerable", True, ["c4916ac6a236-01012"]),
    ("A059", "哪些电动车不能进校园？", "answerable", True, ["c4916ac6a236-01013"]),
    ("A060", "电动车违规会怎么处理？", "answerable", True, ["c4916ac6a236-01015"]),
    ("A061", "电动车违规如何举报？", "answerable", True, ["c4916ac6a236-01017"]),
    ("A062", "校内机动车限速多少？", "answerable", True, ["96e94b03dd1f-01060"]),
    ("A063", "校内停车如何收费？", "answerable", True, ["96e94b03dd1f-01061"]),
    ("A064", "进出校园的车辆要遵守什么规定？", "answerable", True, ["a9b1ca428a9e-01030"]),
    # ---------- answerable：医疗 ----------
    ("A065", "学生医保如何缴费？", "answerable", True, ["6c84b710c0d1-01438"]),
    ("A066", "学生如何就医？", "answerable", True, ["6c84b710c0d1-01442"]),
    ("A067", "低保家庭学生医保有什么减免？", "answerable", True, ["6c84b710c0d1-01441"]),
    # ---------- answerable：留学生 / 海外交流 ----------
    ("A068", "学生可以参加海外交流项目吗？", "answerable", True, ["8beaca14b3b6-00087"]),
    ("A069", "学校海外项目的目的是什么？", "answerable", True, ["0a91830dad84-02645"]),
    ("A070", "留学生如何选课？", "answerable", True, ["6663ef050456-01310"]),
    ("A071", "留学生日常教学如何管理？", "answerable", True, ["24a1c510a9fc-00688"]),
    ("A072", "留学生校外住宿要办什么手续？", "answerable", True, ["9c697a539b03-01403"]),
    ("A073", "港澳台学生医疗保障如何？", "answerable", True, ["0a91830dad84-02582"]),
    # ---------- answerable：选课 / 创新创业 / 其他 ----------
    ("A074", "选课前需要做什么准备？", "answerable", True, ["8beaca14b3b6-00169"]),
    ("A075", "选课时需要注意什么？", "answerable", True, ["8beaca14b3b6-00170"]),
    ("A076", "大学生创新创业项目执行期多久？", "answerable", True, ["8beaca14b3b6-00534"]),
    ("A077", "科创活动可以申请课程免修吗？", "answerable", True, ["bea32bdf2575-00670"]),
    ("A078", "创新创业基地安全管理有什么要求？", "answerable", True, ["89dede0d1367-03569"]),
    ("A079", "实验室使用加热设备有什么要求？", "answerable", True, ["6b9811c9b56c-03756"]),
    ("A080", "哪些行为属于科研失信？", "answerable", True, ["a0c44f743631-02284"]),
    ("A081", "学生在校期间享有哪些权利？", "answerable", True, ["8beaca14b3b6-00009"]),
    ("A082", "学生在校期间需要履行哪些义务？", "answerable", True, ["8beaca14b3b6-00010"]),
    ("A083", "课程考核成绩如何登录？", "answerable", True, ["8beaca14b3b6-00218"]),
    ("A084", "留学生在校外住宿要遵守什么规定？", "answerable", True, ["9c697a539b03-01405"]),
    ("A085", "学校如何开展学生诚信教育？", "answerable", True, ["8beaca14b3b6-00022"]),
    ("A086", "学校在勤工助学中的职责是什么？", "answerable", True, ["48db9eefd40f-01177"]),
    ("A087", "学生认为权益受侵害可以向哪里申诉？", "answerable", True, ["8beaca14b3b6-00062"]),
    ("A088", "学生必须参加哪些课程的考核？", "answerable", True, ["16f8710dd84e-00710"]),
    ("A089", "毕业设计答辩成绩如何评定？", "answerable", True, ["a22e25d1efc4-00818"]),
    ("A090", "辅修专业缺课会有什么后果？", "answerable", True, ["7c65d481c9ca-00864"]),
    ("A091", "转专业申请材料有什么要求？", "answerable", True, ["8beaca14b3b6-00127"]),
    ("A092", "考试中严重违规会怎么处理？", "answerable", True, ["8beaca14b3b6-00234"]),
    ("A093", "考试违规认定处理的依据是什么？", "answerable", True, ["8beaca14b3b6-00226"]),
    ("A094", "考试作弊有哪些具体情形？", "answerable", True, ["e8f90300cd2c-01293"]),
    ("A095", "电动车停放场地如何管理？", "answerable", True, ["c4916ac6a236-01014"]),
    ("A096", "校园公共场所禁止哪些行为？", "answerable", True, ["a9b1ca428a9e-01040"]),
    ("A097", "创新创业项目经费如何管理？", "answerable", True, ["8beaca14b3b6-00535"]),
    ("A098", "校内大型活动如何安全管理？", "answerable", True, ["a9b1ca428a9e-01033"]),
    ("A099", "学生公寓厅长、层长的职责是什么？", "answerable", True, ["b0951330395c-01208"]),
    ("A100", "学生医疗报销如何办理？", "answerable", True, ["0a91830dad84-02863"]),
    ("A101", "港澳台学生如何申请插班就读？", "answerable", True, ["0a91830dad84-02571"]),

    # ---------- multi-hop：跨条款整合 ----------
    ("M001", "学生申请休学后，复学时如何编入班级？", "multi-hop", True, ["8beaca14b3b6-00027", "8beaca14b3b6-00105"]),
    ("M002", "退役士兵复学后可以转专业吗？", "multi-hop", True, ["8beaca14b3b6-00136", "8beaca14b3b6-00139"]),
    ("M003", "学生考试作弊会有什么后果，如何申诉？", "multi-hop", True, ["8beaca14b3b6-00081", "8beaca14b3b6-00059"]),
    ("M004", "新生入学后如何注册，家庭经济困难怎么办？", "multi-hop", True, ["8beaca14b3b6-00015", "8beaca14b3b6-00122"]),
    ("M005", "学生转专业后还能辅修其他专业吗？", "multi-hop", True, ["8beaca14b3b6-00097", "8beaca14b3b6-00018"]),
    ("M006", "宿舍区消防安全和禁止行为分别有哪些规定？", "multi-hop", True, ["ea668a3c48a6-01001", "ea668a3c48a6-01004"]),
    ("M007", "校内电动车行驶和停放充电有什么规定？", "multi-hop", True, ["c4916ac6a236-01012", "c4916ac6a236-01015"]),
    ("M008", "休学创业的学生，其创新创业成果如何折算学分？", "multi-hop", True, ["8beaca14b3b6-00028", "8beaca14b3b6-00019"]),
    ("M009", "学生毕业和结业分别需要满足什么条件？", "multi-hop", True, ["8beaca14b3b6-00033", "8beaca14b3b6-00034"]),
    ("M010", "家庭经济困难学生如何注册并申请资助？", "multi-hop", True, ["8beaca14b3b6-00122", "0a91830dad84-02746"]),
    ("M011", "考试违纪学生还能重修吗，重修成绩如何记载？", "multi-hop", True, ["8beaca14b3b6-00233", "8beaca14b3b6-00020"]),
    ("M012", "留学生选课和日常教学管理有什么规定？", "multi-hop", True, ["6663ef050456-01310", "24a1c510a9fc-00688"]),
    ("M013", "学生医保缴费和就医流程分别是什么？", "multi-hop", True, ["6c84b710c0d1-01438", "6c84b710c0d1-01442"]),
    ("M014", "图书馆借书权限和违规处理分别是什么？", "multi-hop", True, ["76af297ad0fe-03660", "0a91830dad84-02799"]),
    ("M015", "学生转学和转专业分别有什么条件和限制？", "multi-hop", True, ["8beaca14b3b6-00024", "8beaca14b3b6-00023"]),

    # ---------- unanswerable：规章外 / 任务型 / 无依据 ----------
    ("U001", "学校附近哪家餐厅最好吃？", "unanswerable", False, []),
    ("U002", "明天上海会下雨吗？", "unanswerable", False, []),
    ("U003", "帮我写一封求职简历。", "unanswerable", False, []),
    ("U004", "推荐一部好看的电影。", "unanswerable", False, []),
    ("U005", "学校今年的本科毕业生平均起薪是多少？", "unanswerable", False, []),
    ("U006", "请列出本校附近三家评价最高的火锅店，并给出地址。", "unanswerable", False, []),
    ("U007", "能不能帮我写一封向辅导员请假的邮件？", "unanswerable", False, []),
    ("U008", "请用英文详细介绍本校的奖学金种类和评选标准。", "unanswerable", False, []),
    ("U009", "学校党委书记的个人履历是什么？", "unanswerable", False, []),
    ("U010", "明年学校的招生分数线会是多少？", "unanswerable", False, []),
    ("U011", "学校有没有规定老师必须在 24 小时内回复学生邮件？", "unanswerable", False, []),
    ("U012", "帮我写一篇 2000 字的课程论文。", "unanswerable", False, []),
    ("U013", "学校食堂的招牌菜是什么？", "unanswerable", False, []),
    ("U014", "学校是否允许学生在宿舍饲养导盲犬等工作犬？如果允许，有哪些额外要求？", "unanswerable", False, []),
    ("U015", "今天中午食堂有什么菜？", "unanswerable", False, []),
    ("U016", "学校周边哪里可以租到便宜的房子？", "unanswerable", False, []),
    ("U017", "请用英文说明图书馆借阅规则。", "unanswerable", False, []),
    ("U018", "帮我翻译一下这段话：校园制度问答助手。", "unanswerable", False, []),
    ("U019", "学校哪个专业就业前景最好？", "unanswerable", False, []),
    ("U020", "老师个人微信号是多少？", "unanswerable", False, []),
    ("U021", "学校 WiFi 密码是什么？", "unanswerable", False, []),
    ("U022", "帮我查一下明天从上海到北京的高铁时刻表。", "unanswerable", False, []),
    ("U023", "学校哪家奶茶店最好喝？", "unanswerable", False, []),
    ("U024", "请用英文介绍本校的历史沿革。", "unanswerable", False, []),
    ("U025", "学校有没有规定学生必须几点睡觉？", "unanswerable", False, []),
    ("U026", "帮我做一份这学期的学习计划表。", "unanswerable", False, []),
    ("U027", "学校附近有什么好玩的景点？", "unanswerable", False, []),
    ("U028", "请翻译成英文：缓考申请最晚什么时候交？", "unanswerable", False, []),
    ("U029", "今年国家公务员考试时间是什么时候？", "unanswerable", False, []),
    ("U030", "帮我写一份入党申请书。", "unanswerable", False, []),
    ("U031", "帮我写一首诗。", "unanswerable", False, []),
    ("U032", "学校明年学费会涨吗？", "unanswerable", False, []),
    ("U033", "哪个老师讲课最好？", "unanswerable", False, []),
    ("U034", "学校附近哪里可以健身？", "unanswerable", False, []),
    ("U035", "帮我查一下快递到哪了。", "unanswerable", False, []),
    ("U036", "学校有没有规定学生不能谈恋爱？", "unanswerable", False, []),

    # ---------- ambiguous：过短无具体指向 ----------
    ("B001", "奖学金怎么拿到？", "ambiguous", False, []),
    ("B002", "助学金怎么办？", "ambiguous", False, []),
    ("B003", "处分怎么消除？", "ambiguous", False, []),
    ("B004", "学分不够怎么办？", "ambiguous", False, []),
    ("B005", "补助怎么申请？", "ambiguous", False, []),
    ("B006", "评优怎么评？", "ambiguous", False, []),
    ("B007", "加分怎么加？", "ambiguous", False, []),
    ("B008", "补贴有多少？", "ambiguous", False, []),
    ("B009", "资助怎么拿？", "ambiguous", False, []),
    ("B010", "处分怎么算？", "ambiguous", False, []),
]


def main() -> None:
    chunks = load_chunks()
    out = []
    for qid, question, category, should_answer, gold_ids in ITEMS:
        item = {
            "id": qid,
            "question": question,
            "category": category,
            "should_answer": should_answer,
            "source_hint": category,
        }
        if gold_ids:
            gold = []
            for cid in gold_ids:
                r = chunks.get(cid)
                if r is None:
                    raise SystemExit(f"chunk not found: {cid}")
                section = (r.get("section") or "").strip()
                gold.append({
                    "document_id": r.get("doc_id"),
                    "snippet": (r.get("text") or "").strip(),
                    "page": _coerce_page(r.get("page")),
                    "section": section or (r.get("title") or "").strip(),
                })
            item["gold"] = gold
        out.append(item)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    n_ans = sum(1 for i in out if i["should_answer"])
    n_gold = sum(1 for i in out if i["should_answer"] and i.get("gold"))
    print(f"total: {len(out)}")
    print(f"answerable+multi-hop: {n_ans}, 其中含 gold: {n_gold} ({n_gold/max(n_ans,1):.0%})")
    for cat in ("answerable", "multi-hop", "unanswerable", "ambiguous"):
        print(f"  {cat}: {sum(1 for i in out if i['category']==cat)}")


if __name__ == "__main__":
    main()

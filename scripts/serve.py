"""最小 HTTP 服务与极简 Web 页面。

实现目标（成员 D：前端与接口负责人）：
- 提供 `POST /api/answer` JSON API，按 `docs/web_api_contract.md` 契约返回
  `AnswerResult` 的 JSON 表示；
- 提供一个单页 Web 界面：多轮对话形态、答案 Markdown 轻量渲染、
  引用条款卡片化（点击卡片高亮正文锚点）、追问 chips、拒答提示；
- 监控 P95 响应时延：`GET /api/stats` 返回进程内累计的时延统计，
  对应 `docs/requirements.md` 的「基础档 ≤8s / 进阶档 ≤5s」目标；
- 提供 `GET /health` 存活探针，供进程守护与健康检查使用。

依赖约束：
- 只使用 Python 标准库和现有包（`campus_qa.service` 等），不额外引入 Web 框架；
- 前端为单文件 vanilla JS + CSS，不引框架、不引 CDN（离线可用）；
- 前端只展示后端 `AnswerResult` 内容或做 `refuse_reason` 固定映射，不自行生成答案；
- 所有用户/模型内容通过 textContent 拼 DOM，禁止 innerHTML 拼接（防注入）。
"""

from __future__ import annotations

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from campus_qa.config import settings
from campus_qa.service import answer_question


# 响应时延目标（秒），对应 docs/requirements.md 非功能需求。
LATENCY_TARGET_BASE_SECONDS = 8.0
LATENCY_TARGET_ADVANCED_SECONDS = 5.0

# 时延统计保留的最大样本数，超出后丢弃最旧样本。
_MAX_LATENCY_SAMPLES = 1000


PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>校园制度问答助手</title>
    <style>
      :root {
        --primary: #2563eb;
        --primary-hover: #1d4ed8;
        --primary-soft: #eff6ff;
        --border: #d1d5db;
        --border-soft: #e5e7eb;
        --bg: #f5f7fa;
        --card-bg: #ffffff;
        --text: #1f2329;
        --text-soft: #374151;
        --muted: #6b7280;
        --faint: #9ca3af;
        --highlight: #fef08a;
        --danger: #b91c1c;
        --radius: 12px;
        --radius-sm: 8px;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        background: var(--bg);
        display: flex;
        justify-content: center;
        align-items: flex-start;
        padding: 1.5rem 1rem 3rem;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;
        color: var(--text);
      }
      .container { width: 100%; max-width: 760px; display: flex; flex-direction: column; }
      header { text-align: center; margin-bottom: 1rem; }
      h1 { font-size: 1.6rem; margin: 0 0 0.4rem; }
      .subtitle { color: var(--muted); font-size: 0.95rem; margin: 0; }

      .search-box { display: flex; gap: 0.5rem; margin-bottom: 0.4rem; }
      input[type=text] {
        flex: 1;
        padding: 0.8rem 1rem;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        font-size: 1rem;
        background: var(--card-bg);
        color: var(--text);
      }
      input[type=text]:focus { outline: none; border-color: var(--primary); }
      button#ask {
        padding: 0.8rem 1.4rem;
        border: none;
        border-radius: var(--radius);
        background: var(--primary);
        color: #fff;
        font-size: 1rem;
        cursor: pointer;
      }
      button#ask:hover { background: var(--primary-hover); }
      button#ask:disabled { background: var(--faint); cursor: not-allowed; }

      .section-label { color: var(--muted); font-size: 0.82rem; margin: 0.5rem 0 0.4rem; }
      .chips { display: flex; flex-wrap: wrap; gap: 0.5rem; }
      .suggestions-chips { justify-content: center; }
      .chip {
        padding: 0.4rem 0.9rem;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: var(--card-bg);
        color: var(--text-soft);
        font-size: 0.85rem;
        cursor: pointer;
      }
      .chip:hover { border-color: var(--primary); color: var(--primary); }

      .status { color: var(--muted); min-height: 1.3rem; margin: 0.5rem 0; font-size: 0.9rem; }
      .status.loading::before {
        content: '';
        display: inline-block;
        width: 0.8rem;
        height: 0.8rem;
        margin-right: 0.45rem;
        border: 2px solid var(--border);
        border-top-color: var(--primary);
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        vertical-align: -1px;
      }
      @keyframes spin { to { transform: rotate(360deg); } }

      .conversation { display: flex; flex-direction: column; gap: 1.1rem; margin-top: 0.75rem; }
      .round { display: flex; flex-direction: column; gap: 0.55rem; }
      .bubble.user {
        align-self: flex-end;
        max-width: 85%;
        background: var(--primary);
        color: #fff;
        border-radius: var(--radius) var(--radius) 4px var(--radius);
        padding: 0.55rem 1rem;
        font-size: 0.95rem;
        word-break: break-word;
        text-align: left;
      }
      .bubble.assistant {
        align-self: flex-start;
        width: 100%;
        display: flex;
        flex-direction: column;
        gap: 0.6rem;
      }

      .answer {
        background: var(--card-bg);
        border: 1px solid var(--border-soft);
        border-radius: var(--radius);
        padding: 0.9rem 1.1rem;
        line-height: 1.75;
        font-size: 0.95rem;
        text-align: left;
      }
      .answer.refused { color: var(--muted); background: #fafafa; }
      .answer .md-para { margin: 0 0 0.55rem; }
      .answer .md-para:last-child { margin-bottom: 0; }
      .answer .md-heading { font-weight: 600; font-size: 1rem; margin: 0.6rem 0 0.3rem; }
      .answer ul, .answer ol { margin: 0.3rem 0 0.55rem; padding-left: 1.4rem; }
      .answer li { margin: 0.2rem 0; }
      .answer strong { font-weight: 600; }
      .answer code {
        background: #f3f4f6;
        border: 1px solid var(--border-soft);
        border-radius: 4px;
        padding: 0.05rem 0.35rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.88em;
      }
      sup.cite-anchor { color: var(--primary); cursor: pointer; font-weight: 600; margin: 0 1px; font-size: 0.8em; }
      sup.cite-anchor:hover { text-decoration: underline; }
      sup.cite-anchor.active { background: var(--highlight); border-radius: 3px; }

      .citations { display: flex; flex-direction: column; gap: 0.5rem; }
      .citations > strong { color: var(--muted); font-size: 0.85rem; }
      .citation-card {
        background: var(--card-bg);
        border: 1px solid var(--border-soft);
        border-radius: var(--radius-sm);
        padding: 0.65rem 0.85rem;
        text-align: left;
        transition: border-color 0.15s, background-color 0.15s;
      }
      .citation-card.clickable { cursor: pointer; }
      .citation-card.clickable:hover { border-color: var(--primary); }
      .citation-card.active { border-color: var(--primary); background: var(--highlight); }
      .citation-card-head { display: flex; justify-content: space-between; align-items: baseline; gap: 0.5rem; margin-bottom: 0.3rem; }
      .citation-card-title { font-weight: 600; font-size: 0.9rem; color: var(--text); }
      .citation-badge { color: var(--primary); font-weight: 600; font-size: 0.8rem; white-space: nowrap; }
      .citation-snippet { font-size: 0.85rem; color: var(--text-soft); line-height: 1.6; word-break: break-word; }
      .citation-toggle { margin-top: 0.3rem; padding: 0; background: none; border: none; color: var(--primary); font-size: 0.8rem; cursor: pointer; }
      .citation-card-foot {
        display: flex; justify-content: space-between; align-items: center; gap: 0.5rem;
        margin-top: 0.4rem; font-size: 0.78rem; color: var(--muted); flex-wrap: wrap;
      }
      .citation-card-foot a { color: var(--primary); text-decoration: none; white-space: nowrap; }
      .citation-card-foot a:hover { text-decoration: underline; }

      .clarify { margin-top: 0.25rem; }
      .meta { font-size: 0.8rem; color: var(--muted); }
      .followup { margin-top: 0.25rem; }

      .toolbar { display: flex; justify-content: flex-end; margin-top: 1rem; }
      button.ghost {
        background: none; border: 1px solid var(--border); border-radius: var(--radius-sm);
        color: var(--muted); padding: 0.4rem 0.9rem; font-size: 0.85rem; cursor: pointer;
      }
      button.ghost:hover { color: var(--danger); border-color: var(--danger); }

      .footer { margin-top: 1.5rem; text-align: center; font-size: 0.78rem; color: var(--faint); }

      @media (max-width: 640px) {
        body { padding: 1rem 0.75rem 2rem; }
        .search-box { flex-direction: column; }
        input[type=text] { width: 100%; }
        .chips { flex-wrap: nowrap; overflow-x: auto; justify-content: flex-start; padding-bottom: 0.25rem; -webkit-overflow-scrolling: touch; }
        .chip { flex: 0 0 auto; }
        .bubble.user { max-width: 92%; }
        .citation-card { width: 100%; }
      }
    </style>
  </head>
  <body>
    <div class="container">
      <header>
        <h1>校园制度问答助手</h1>
        <p class="subtitle">缓考、休学、选课、奖助学金、借阅……校内制度问题都可以问我</p>
      </header>

      <div class="search-box">
        <input id="question" type="text" placeholder="比如：缓考申请最晚什么时候交？" autocomplete="off" />
        <button id="ask" type="button">提问</button>
      </div>

      <div class="section-label">可以试试这样问</div>
      <div class="chips suggestions-chips" id="suggestions"></div>

      <div id="status" class="status"></div>
      <div id="conversation" class="conversation"></div>
      <div id="followup" class="followup"></div>

      <div class="toolbar">
        <button id="clear" type="button" class="ghost">清空对话</button>
      </div>

      <div class="footer">本系统回答仅供参考，具体以学校正式制度文件为准。</div>
    </div>

    <script>
      const questionInput = document.getElementById('question');
      const askButton = document.getElementById('ask');
      const clearButton = document.getElementById('clear');
      const statusEl = document.getElementById('status');
      const conversationEl = document.getElementById('conversation');
      const followupEl = document.getElementById('followup');

      const SUGGESTIONS = [
        '缓考申请最晚什么时候交？',
        '申请休学要满足哪些条件？',
        '图书馆能借几本书？',
        '宿舍可以用大功率电器吗？',
        '申请国家助学金需要哪些条件？',
      ];

      const FOLLOWUPS = [
        { keys: ['缓考', '补考', '考试', '考核'], qs: ['补考课程冲突怎么处理？', '缓考申请需要提交哪些材料？', '考试作弊会受到什么处分？'] },
        { keys: ['休学', '退学', '复学', '学籍'], qs: ['休学后宿舍和学费怎么处理？', '申请休学要走哪些审批流程？', '学生可以申请提前毕业吗？'] },
        { keys: ['转专业'], qs: ['转专业的申请时间和基本条件是什么？', '转专业后已修学分怎么认定？'] },
        { keys: ['选课', '学分', '退选', '退课'], qs: ['本科生每学期最多能修多少学分？', '退选课程需要满足什么条件？'] },
        { keys: ['奖学金', '助学金', '资助', '勤工助学'], qs: ['申请国家助学金需要哪些条件？', '奖学金怎么评选？'] },
        { keys: ['宿舍', '住宿', '宠物', '电器'], qs: ['宿舍可以使用大功率电器吗？', '学校允许在宿舍养宠物吗？'] },
        { keys: ['图书馆', '借书', '借阅', '图书'], qs: ['图书馆能借几本书？', '图书逾期未还会怎么处理？'] },
        { keys: ['军训'], qs: ['新生军训缺席的处理办法是什么？', '军训可以申请免训吗？'] },
        { keys: ['处分', '违纪', '作弊'], qs: ['学校对考试作弊的处分有哪些等级？', '学生违纪处分有哪些种类？'] },
        { keys: ['兼职', '打工'], qs: ['学校对学生校外兼职有什么规定？'] },
        { keys: ['毕业'], qs: ['学生可以申请提前毕业吗？', '补办学历学位证明需要什么流程？'] },
      ];

      const DEFAULT_FOLLOWUPS = [
        '缓考申请最晚什么时候交？',
        '本科生每学期最多能修多少学分？',
        '图书馆能借几本书？',
        '申请国家助学金需要哪些条件？',
      ];

      let roundCounter = 0;

      function refuseMessage(code) {
        switch (code) {
          case 'empty_question':
            return '请输入具体的校园制度问题。';
          case 'unsupported_language':
            return '目前仅支持中文提问，请改用中文。';
          case 'ambiguous_question':
            return '问题信息不足，请补充具体制度名称或场景后重新提问。';
          case 'retrieval_empty':
            return '暂未找到相关制度条款，请确认问题表述。';
          case 'retrieval_low_score':
            return '检索匹配度较低，暂无法给出可靠回答。';
          case 'invalid_model_output':
            return '内部处理出现问题，请稍后重试。';
          case 'no_valid_citation':
            return '当前回答缺少有效引用，已拒绝作答。';
          case 'no_evidence_in_corpus':
            return '现有制度文件无法支撑该问题，请咨询辅导员或相关部门。';
          default:
            return '当前问题暂无法回答，请咨询相关老师。';
        }
      }

      function followUpSuggestions(q) {
        for (let i = 0; i < FOLLOWUPS.length; i++) {
          const group = FOLLOWUPS[i];
          const hit = group.keys.some(function (k) { return q.indexOf(k) !== -1; });
          if (hit) return group.qs;
        }
        return DEFAULT_FOLLOWUPS;
      }

      function renderChips(container, questions) {
        container.innerHTML = '';
        (questions || []).forEach(function (q) {
          const chip = document.createElement('button');
          chip.type = 'button';
          chip.className = 'chip';
          chip.textContent = q;
          chip.addEventListener('click', function () { askQuestion(q); });
          container.appendChild(chip);
        });
      }

      function askQuestion(q) {
        questionInput.value = q;
        ask();
      }

      function makeCiteMap(citations) {
        const map = {};
        (citations || []).forEach(function (c) {
          if (typeof c.citation_index === 'number') map[c.citation_index] = true;
        });
        return map;
      }

      function appendSup(container, n, roundId) {
        const sup = document.createElement('sup');
        sup.className = 'cite-anchor';
        sup.textContent = '[' + n + ']';
        sup.setAttribute('data-cite', n);
        sup.setAttribute('data-round', roundId);
        sup.title = '查看引用条款 ' + n;
        sup.addEventListener('click', function () { highlightCitation(roundId, n); });
        container.appendChild(sup);
      }

      function highlightCitation(roundId, n) {
        const target = document.getElementById('cite-' + roundId + '-' + n);
        if (!target) return;
        document.querySelectorAll('sup.cite-anchor.active, .citation-card.active').forEach(function (el) {
          el.classList.remove('active');
        });
        document.querySelectorAll('sup.cite-anchor[data-round="' + roundId + '"][data-cite="' + n + '"]').forEach(function (el) {
          el.classList.add('active');
        });
        target.classList.add('active');
        target.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }

      function renderInline(container, text, citeMap, roundId) {
        const tokenRe = /(\*\*[^*]+\*\*|`[^`]+`|\[\d+\])/;
        let remaining = text || '';
        while (remaining.length > 0) {
          const match = remaining.match(tokenRe);
          if (!match) {
            container.appendChild(document.createTextNode(remaining));
            break;
          }
          const idx = match.index;
          if (idx > 0) container.appendChild(document.createTextNode(remaining.slice(0, idx)));
          const token = match[0];
          if (token.indexOf('**') === 0) {
            const strong = document.createElement('strong');
            renderInline(strong, token.slice(2, -2), citeMap, roundId);
            container.appendChild(strong);
          } else if (token.indexOf('`') === 0) {
            const code = document.createElement('code');
            code.textContent = token.slice(1, -1);
            container.appendChild(code);
          } else {
            const n = token.slice(1, -1);
            if (Object.prototype.hasOwnProperty.call(citeMap, n)) {
              appendSup(container, n, roundId);
            } else {
              container.appendChild(document.createTextNode(token));
            }
          }
          remaining = remaining.slice(idx + token.length);
        }
      }

      function renderMarkdown(container, text, citations, roundId) {
        container.textContent = '';
        const citeMap = makeCiteMap(citations);
        const lines = (text || '').split('\n');
        let listEl = null;
        let listType = null;
        function closeList() { listEl = null; listType = null; }

        lines.forEach(function (line) {
          const bullet = line.match(/^\s*[-*]\s+(.*)$/);
          const ordered = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
          const heading = line.match(/^\s*(#{1,6})\s+(.*)$/);

          if (bullet) {
            if (listType !== 'ul') { closeList(); listEl = document.createElement('ul'); listType = 'ul'; container.appendChild(listEl); }
            const li = document.createElement('li');
            renderInline(li, bullet[1], citeMap, roundId);
            listEl.appendChild(li);
          } else if (ordered) {
            if (listType !== 'ol') { closeList(); listEl = document.createElement('ol'); listType = 'ol'; container.appendChild(listEl); }
            const li = document.createElement('li');
            renderInline(li, ordered[2], citeMap, roundId);
            listEl.appendChild(li);
          } else if (heading) {
            closeList();
            const h = document.createElement('div');
            h.className = 'md-heading';
            renderInline(h, heading[2], citeMap, roundId);
            container.appendChild(h);
          } else {
            closeList();
            if (line.trim() === '') return;
            const p = document.createElement('div');
            p.className = 'md-para';
            renderInline(p, line, citeMap, roundId);
            container.appendChild(p);
          }
        });
      }

      function renderCitationCard(c, roundId) {
        const card = document.createElement('div');
        card.className = 'citation-card';
        if (typeof c.citation_index === 'number') {
          card.id = 'cite-' + roundId + '-' + c.citation_index;
          card.classList.add('clickable');
        }

        const head = document.createElement('div');
        head.className = 'citation-card-head';
        const title = document.createElement('span');
        title.className = 'citation-card-title';
        title.textContent = c.section || c.document_id || '制度条款';
        head.appendChild(title);
        if (typeof c.citation_index === 'number') {
          const badge = document.createElement('span');
          badge.className = 'citation-badge';
          badge.textContent = '[' + c.citation_index + ']';
          head.appendChild(badge);
        }
        card.appendChild(head);

        const snippet = document.createElement('div');
        snippet.className = 'citation-snippet';
        const full = c.snippet || '';
        const MAX = 160;
        snippet.textContent = full.length > MAX ? full.slice(0, MAX) + '…' : full;
        card.appendChild(snippet);
        if (full.length > MAX) {
          const toggle = document.createElement('button');
          toggle.type = 'button';
          toggle.className = 'citation-toggle';
          toggle.textContent = '展开';
          toggle.addEventListener('click', function (ev) {
            ev.stopPropagation();
            const expanded = toggle.textContent === '收起';
            snippet.textContent = expanded ? full.slice(0, MAX) + '…' : full;
            toggle.textContent = expanded ? '展开' : '收起';
          });
          card.appendChild(toggle);
        }

        const foot = document.createElement('div');
        foot.className = 'citation-card-foot';
        const parts = [];
        if (typeof c.score === 'number') parts.push('score ' + c.score.toFixed(2));
        if (typeof c.page === 'number') parts.push('page ' + c.page);
        if (c.section) parts.push(c.section);
        if (parts.length) {
          const meta = document.createElement('span');
          meta.textContent = parts.join(' · ');
          foot.appendChild(meta);
        }
        if (c.url) {
          const link = document.createElement('a');
          link.href = c.url;
          link.target = '_blank';
          link.rel = 'noopener noreferrer';
          link.textContent = '查看原文';
          foot.appendChild(link);
        }
        card.appendChild(foot);

        if (typeof c.citation_index === 'number') {
          card.addEventListener('click', function () { highlightCitation(roundId, c.citation_index); });
        }

        return card;
      }

      function buildClarifyBlock(text) {
        const block = document.createElement('div');
        block.className = 'clarify';
        const label = document.createElement('div');
        label.className = 'section-label';
        label.textContent = '追问澄清';
        block.appendChild(label);
        const chips = document.createElement('div');
        chips.className = 'chips';
        block.appendChild(chips);
        renderChips(chips, [text]);
        return block;
      }

      function renderFollowups(q, backendQuestions) {
        followupEl.innerHTML = '';
        const label = document.createElement('div');
        label.className = 'section-label';
        label.textContent = '你可能还想问';
        followupEl.appendChild(label);
        const chips = document.createElement('div');
        chips.className = 'chips';
        followupEl.appendChild(chips);
        const hasBackend = Array.isArray(backendQuestions) && backendQuestions.length > 0;
        renderChips(chips, hasBackend ? backendQuestions : followUpSuggestions(q));
      }

      function appendRound(question, data) {
        const rid = 'r' + (++roundCounter);

        const round = document.createElement('div');
        round.className = 'round';

        const userBubble = document.createElement('div');
        userBubble.className = 'bubble user';
        userBubble.textContent = question;
        round.appendChild(userBubble);

        const assistantBubble = document.createElement('div');
        assistantBubble.className = 'bubble assistant';

        const answerEl = document.createElement('div');
        answerEl.className = 'answer';
        if (data.refused) {
          answerEl.classList.add('refused');
          answerEl.textContent = refuseMessage(data.refuse_reason) || (data.answer || '当前问题暂无法回答。');
        } else {
          renderMarkdown(answerEl, data.answer || '系统未返回答案。', data.citations, rid);
        }
        assistantBubble.appendChild(answerEl);

        if (Array.isArray(data.citations) && data.citations.length > 0) {
          const citeSection = document.createElement('div');
          citeSection.className = 'citations';
          const heading = document.createElement('strong');
          heading.textContent = '引用条款';
          citeSection.appendChild(heading);
          data.citations.forEach(function (c) {
            citeSection.appendChild(renderCitationCard(c, rid));
          });
          assistantBubble.appendChild(citeSection);
        }

        const isAmbiguous = data.refused && data.refuse_reason === 'ambiguous_question';
        if (isAmbiguous && data.suggested_contact) {
          assistantBubble.appendChild(buildClarifyBlock(data.suggested_contact));
        }

        const metaParts = [];
        if (typeof data.confidence === 'number') metaParts.push('系统置信度：' + data.confidence.toFixed(2));
        if (typeof data.top_retrieval_score === 'number') metaParts.push('检索匹配度：' + data.top_retrieval_score.toFixed(2));
        if (data.suggested_contact && !isAmbiguous) metaParts.push('建议：' + data.suggested_contact);
        if (metaParts.length) {
          const metaEl = document.createElement('div');
          metaEl.className = 'meta';
          metaEl.textContent = metaParts.join('  |  ');
          assistantBubble.appendChild(metaEl);
        }

        round.appendChild(assistantBubble);
        conversationEl.appendChild(round);

        renderFollowups(question, data.follow_up_questions);
      }

      async function ask() {
        const q = (questionInput.value || '').trim();
        if (!q) {
          statusEl.textContent = '请先输入一个问题。';
          return;
        }

        askButton.disabled = true;
        statusEl.textContent = '正在查询制度条款，请稍候……';
        statusEl.classList.add('loading');

        try {
          const resp = await fetch('/api/answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: q })
          });

          if (!resp.ok) {
            let msg = null;
            try {
              const errData = await resp.json();
              if (errData && errData.message) msg = errData.message;
            } catch (e) { /* 忽略错误体解析失败 */ }
            if (resp.status >= 500) msg = msg || '服务暂时不可用，请稍后重试。';
            throw new Error(msg || ('请求失败（HTTP ' + resp.status + '）'));
          }

          const data = await resp.json();
          statusEl.textContent = '';
          statusEl.classList.remove('loading');

          appendRound(q, data);
          questionInput.value = '';
        } catch (err) {
          console.error(err);
          const isNetwork = (err instanceof TypeError);
          statusEl.textContent = isNetwork
            ? '网络异常，请检查服务是否已启动。'
            : ((err && err.message) || '服务暂时不可用，请稍后重试。');
        } finally {
          statusEl.classList.remove('loading');
          askButton.disabled = false;
        }
      }

      function clearConversation() {
        conversationEl.innerHTML = '';
        followupEl.innerHTML = '';
        roundCounter = 0;
        statusEl.textContent = '';
        statusEl.classList.remove('loading');
      }

      askButton.addEventListener('click', ask);
      clearButton.addEventListener('click', clearConversation);
      questionInput.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter') ask();
      });

      renderChips(document.getElementById('suggestions'), SUGGESTIONS);
    </script>
  </body>
</html>
"""


def _percentile(sorted_values: List[float], p: float) -> float:
    """最近秩（nearest-rank）百分位数。

    输入要求已升序排列；空列表返回 0.0。
    """

    if not sorted_values:
        return 0.0
    rank = int(math.ceil(p * len(sorted_values)))
    index = max(0, min(len(sorted_values) - 1, rank - 1))
    return sorted_values[index]


class LatencyTracker:
    """线程安全的问答时延记录器，用于 P95 监控。

    - `record` 记录一次问答耗时（秒）；
    - `summary` 返回累计样本的统计：count、avg、p50/p95/p99、max，
      以及是否落在基础档 / 进阶档时延目标内。
    """

    def __init__(self, max_samples: int = _MAX_LATENCY_SAMPLES) -> None:
        self._lock = threading.Lock()
        self._latencies: List[float] = []
        self._max_samples = max_samples
        self._started_at = time.monotonic()

    def record(self, seconds: float) -> None:
        with self._lock:
            self._latencies.append(seconds)
            if len(self._latencies) > self._max_samples:
                self._latencies = self._latencies[-self._max_samples:]

    def uptime_seconds(self) -> float:
        return time.monotonic() - self._started_at

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            samples = sorted(self._latencies)
            count = len(samples)
            uptime = time.monotonic() - self._started_at

        if count == 0:
            return {
                "count": 0,
                "uptime_seconds": round(uptime, 3),
                "avg_seconds": None,
                "p50_seconds": None,
                "p95_seconds": None,
                "p99_seconds": None,
                "max_seconds": None,
                "p95_within_base_target": None,
                "p95_within_advanced_target": None,
            }

        avg = sum(samples) / count
        p95 = _percentile(samples, 0.95)
        return {
            "count": count,
            "uptime_seconds": round(uptime, 3),
            "avg_seconds": round(avg, 3),
            "p50_seconds": round(_percentile(samples, 0.50), 3),
            "p95_seconds": round(p95, 3),
            "p99_seconds": round(_percentile(samples, 0.99), 3),
            "max_seconds": round(samples[-1], 3),
            "p95_within_base_target": p95 <= LATENCY_TARGET_BASE_SECONDS,
            "p95_within_advanced_target": p95 <= LATENCY_TARGET_ADVANCED_SECONDS,
        }


# 进程级单例：所有请求共享同一个时延统计。
LATENCY_TRACKER = LatencyTracker()


def _parse_question_payload(raw_body: bytes) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """解析请求体，返回 ``(question, error)``。

    - 合法时返回 ``(question, None)``；
    - 非法 JSON 或非对象时返回 ``(None, error_dict)``，由调用方回 400。
    """

    if raw_body:
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, {"error": "invalid_json", "message": "请求体不是合法 JSON"}
        if not isinstance(payload, dict):
            return None, {"error": "invalid_json", "message": "请求体必须是 JSON 对象"}
    else:
        payload = {}

    raw_question = payload.get("question", "")
    if raw_question is None:
        question = ""
    elif isinstance(raw_question, str):
        question = raw_question
    else:
        # 契约约定 question 必为字符串；其他类型直接 400，避免 str() 转出噪音。
        return None, {"error": "invalid_question", "message": "question 字段必须是字符串"}

    return question, None


def build_answer_payload(question: str) -> Dict[str, Any]:
    """调用问答服务，返回 ``AnswerResult`` 的 JSON 字典。

    抽成独立函数便于单元测试契约序列化，`do_POST` 与测试复用同一路径。
    """

    result = answer_question(question)
    # mode="json" 确保 RefuseReason 枚举等字段按字符串值序列化。
    return result.model_dump(mode="json")


def handle_answer_request(question: str) -> Tuple[int, Dict[str, Any]]:
    """执行问答并返回 ``(status, body)``。

    - 正常返回 ``(200, AnswerResult JSON)``；
    - 服务异常（如检索 / 生成链路意外抛出）返回 ``(500, error JSON)``，
      符合 `docs/web_api_contract.md` 的「服务异常返回 5xx 与错误信息」。
    """

    try:
        return 200, build_answer_payload(question)
    except Exception as exc:  # noqa: BLE001 - 边界兜底，不向客户端泄漏堆栈
        return 500, {"error": "internal_error", "message": "服务暂时不可用，请稍后重试。"}


class QAHandler(BaseHTTPRequestHandler):
    """简单的 HTTP 处理器。

    - GET `/` / `/index.html`：返回单页 HTML；
    - GET `/health`：存活探针；
    - GET `/api/stats`：问答时延统计（P50/P95/P99）；
    - POST `/api/answer`：接收 JSON 请求并返回 `AnswerResult` 的 JSON。
    """

    def do_GET(self) -> None:  # pragma: no cover - 手工测试为主
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            body = PAGE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "campus-policy-qa",
                    "uptime_seconds": round(LATENCY_TRACKER.uptime_seconds(), 3),
                    "index_files_present": (
                        (settings.index_dir / "embeddings.npy").exists()
                        and (settings.index_dir / "manifest.json").exists()
                    ),
                },
            )
            return

        if parsed.path == "/api/stats":
            self._send_json(200, LATENCY_TRACKER.summary())
            return

        self.send_error(404, "Not Found")

    def do_POST(self) -> None:  # pragma: no cover - 手工测试为主
        parsed = urlparse(self.path)
        if parsed.path != "/api/answer":
            self.send_error(404, "Not Found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0

        raw_body = self.rfile.read(length) if length > 0 else b""

        question, error = _parse_question_payload(raw_body)
        if error is not None:
            self._send_json(400, error)
            return

        started = time.monotonic()
        status, body_dict = handle_answer_request(question)
        elapsed = time.monotonic() - started
        LATENCY_TRACKER.record(elapsed)

        self._send_json(status, body_dict)
        self.log_message(
            "answer status=%d elapsed=%.3fs question=%r",
            status,
            elapsed,
            question[:40],
        )

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        response = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover
        # 保留简单的控制台日志，便于本地调试（含时延与问题摘要）。
        super().log_message(format, *args)


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    """启动最小 HTTP 服务。"""

    server = ThreadingHTTPServer((host, port), QAHandler)
    print(f"Serving campus QA on http://{host}:{port}")
    print("  GET /health      存活探针")
    print("  GET /api/stats   问答时延统计（P50/P95/P99）")
    print("  POST /api/answer 问答接口（AnswerResult JSON）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        print("\nServer stopped by user")


if __name__ == "__main__":  # pragma: no cover
    main()

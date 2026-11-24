(() => {
  "use strict";

  const INTENT_LABEL = { policy: "制度", life: "生活", tool: "实时", chitchat: "闲聊" };
  // 整个应用只有一个会话：ID 存在浏览器本地，刷新/重开都接着上次的记录
  const SESSION_KEY = "campus-qa-session";

  const SUGGESTIONS = [
    { icon: "📚", text: "图书馆最多可以借多少本书？" },
    { icon: "📝", text: "缓考怎么申请？" },
    { icon: "🍚", text: "食堂几点开饭？" },
    { icon: "📦", text: "快递驿站在哪里取件？" },
    { icon: "💳", text: "校园卡丢了怎么补办？" },
    { icon: "🌤", text: "现在浦东天气怎么样？" },
    { icon: "📅", text: "本学期第几周了？" },
    { icon: "📞", text: "保卫处值班电话是多少？" },
  ];

  const state = { sessionId: null, sending: false };

  const $ = (id) => document.getElementById(id);
  const chatEl = $("chat");
  const inputEl = $("input");
  const sendBtn = $("send");
  const healthDot = $("health-dot");
  const healthText = $("health-text");
  const themeBtn = $("theme-toggle");
  const clearBtn = $("clear-chat");

  const inner = document.createElement("div");
  inner.className = "chat-inner";
  chatEl.appendChild(inner);

  // ---------------- 基础工具 ----------------

  async function api(path, options = {}) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    const text = await response.text();
    let payload = null;
    try {
      payload = text ? JSON.parse(text) : null;
    } catch (err) {
      payload = null;
    }
    if (!response.ok) {
      const message = (payload && (payload.message || payload.error)) || `HTTP ${response.status}`;
      throw new Error(message);
    }
    return payload;
  }

  function scrollToBottom(force = false) {
    const distance = chatEl.scrollHeight - chatEl.scrollTop - chatEl.clientHeight;
    if (force || distance < 140) {
      chatEl.scrollTop = chatEl.scrollHeight;
    }
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function rememberSession(sessionId) {
    state.sessionId = sessionId;
    try {
      if (sessionId) localStorage.setItem(SESSION_KEY, sessionId);
      else localStorage.removeItem(SESSION_KEY);
    } catch (err) {
      /* 隐私模式下忽略 */
    }
  }

  // ---------------- 单一会话 ----------------

  /** 保证有一个可用会话：优先复用本地记住的，失效则新建。返回历史消息。 */
  async function ensureSession() {
    let stored = null;
    try {
      stored = state.sessionId || localStorage.getItem(SESSION_KEY);
    } catch (err) {
      stored = state.sessionId;
    }

    if (stored) {
      try {
        const data = await api(`/api/sessions/${stored}/messages`);
        state.sessionId = stored;
        return data.messages || [];
      } catch (err) {
        stored = null; // 会话已被删除：重新建一个
      }
    }

    try {
      const data = await api("/api/sessions", { method: "POST", body: JSON.stringify({ title: "校园问答" }) });
      rememberSession(data.session_id);
    } catch (err) {
      console.warn("建会话失败，本次问答按无状态处理", err);
      state.sessionId = null;
    }
    return [];
  }

  // ---------------- 欢迎页 ----------------

  function renderWelcome() {
    inner.innerHTML = "";
    const card = el("div", "welcome");
    card.appendChild(el("div", "welcome-icon", "🎓"));
    card.appendChild(el("h2", null, "有什么校园问题想问？"));
    card.appendChild(
      el(
        "p",
        null,
        "制度类问题会给条款引用，生活类问题会给服务指南出处，天气/校历/电话直接查实时数据。"
      )
    );

    card.appendChild(el("div", "welcome-label", "试试这些问题"));

    const chips = el("div", "chips");
    SUGGESTIONS.forEach((item) => {
      const chip = el("button", "chip");
      chip.type = "button";
      chip.appendChild(el("span", "chip-icon", item.icon));
      chip.appendChild(el("span", "chip-text", item.text));
      chip.addEventListener("click", () => {
        inputEl.value = item.text;
        send();
      });
      chips.appendChild(chip);
    });
    card.appendChild(chips);
    inner.appendChild(card);
  }

  // ---------------- 消息渲染 ----------------

  function intentTag(intent) {
    const label = INTENT_LABEL[intent];
    if (!label) return null;
    return el("span", `tag ${intent}`, label);
  }

  function renderSources(sources) {
    if (!Array.isArray(sources) || sources.length === 0) return null;

    const wrap = el("div", "sources");
    wrap.appendChild(el("div", "sources-label", `引用来源 · ${sources.length}`));
    sources.forEach((source) => {
      const card = el("div", "source-card");
      const head = el("div", "source-head");

      let title = source.title || source.document_id || "来源";
      if (source.kind === "policy") title = `制度 · ${title}`;
      if (source.kind === "life") title = `生活 · ${title}`;
      if (source.kind === "tool") title = `实时 · ${title}`;

      head.appendChild(el("span", "source-title", title));
      card.appendChild(head);

      if (source.url) {
        const link = el("a", "source-link", "查看原文 →");
        link.href = source.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        card.appendChild(link);
      }

      wrap.appendChild(card);
    });
    return wrap;
  }

  function appendUser(text) {
    const wrap = el("div", "msg user");
    const head = el("div", "msg-head");
    head.appendChild(el("span", "avatar", "我"));
    wrap.appendChild(head);
    wrap.appendChild(el("div", "bubble", text));
    inner.appendChild(wrap);
    scrollToBottom(true);
  }

  function appendAssistantShell(intent) {
    const wrap = el("div", "msg assistant");
    const head = el("div", "msg-head");
    head.appendChild(el("span", "avatar", "AI"));
    const tag = intentTag(intent);
    if (tag) head.appendChild(tag);

    const bubble = el("div", "bubble");
    const typing = el("span", "typing");
    typing.appendChild(el("span", "typing-dot"));
    bubble.appendChild(typing);

    wrap.appendChild(head);
    wrap.appendChild(bubble);
    inner.appendChild(wrap);
    scrollToBottom(true);

    return { wrap, head, bubble, intent, body: "" };
  }

  function fillAssistant(shell, text, refused) {
    shell.bubble.classList.toggle("refused", Boolean(refused));
    shell.bubble.textContent = text || "（没有生成内容）";
    shell.body = text || "";
  }

  function attachSources(shell, sources) {
    const node = renderSources(sources);
    if (node) shell.wrap.appendChild(node);
    scrollToBottom();
  }

  function renderHistory(messages) {
    inner.innerHTML = "";
    if (!messages.length) {
      renderWelcome();
      return;
    }
    messages.forEach((message) => {
      if (message.role === "user") {
        appendUser(message.content);
        return;
      }
      const shell = appendAssistantShell(message.intent);
      fillAssistant(shell, message.content, false);
      attachSources(shell, message.sources);
    });
    scrollToBottom(true);
  }

  // ---------------- 发送 ----------------

  function parseEventBlock(block) {
    let event = "message";
    const dataLines = [];
    block.split("\n").forEach((line) => {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    });
    if (!dataLines.length) return null;
    try {
      return { event, data: JSON.parse(dataLines.join("\n")) };
    } catch (err) {
      return null;
    }
  }

  async function send() {
    const text = inputEl.value.trim();
    if (!text || state.sending) return;

    state.sending = true;
    sendBtn.disabled = true;
    inputEl.value = "";
    resizeInput();

    inner.querySelector(".welcome")?.remove();
    appendUser(text);

    const shell = appendAssistantShell(null);

    try {
      await ensureSession();
      const response = await fetch("/api/answer/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, session_id: state.sessionId }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`服务返回 ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let gotDelta = false;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() || "";

        for (const block of blocks) {
          const parsed = parseEventBlock(block);
          if (!parsed) continue;

          if (parsed.event === "meta") {
            shell.intent = parsed.data.intent;
            const tag = intentTag(shell.intent);
            if (tag) shell.head.appendChild(tag);
          } else if (parsed.event === "delta") {
            if (!gotDelta) {
              gotDelta = true;
              shell.bubble.textContent = "";
            }
            shell.bubble.textContent += parsed.data.text || "";
            scrollToBottom();
          } else if (parsed.event === "done") {
            fillAssistant(shell, parsed.data.answer || parsed.data.suggested_contact, parsed.data.refused);
            attachSources(shell, parsed.data.sources);
            shell.head.querySelectorAll(".tag").forEach((node) => node.remove());
            const tag = intentTag(parsed.data.intent);
            if (tag) shell.head.appendChild(tag);
          }
        }
      }

      if (!gotDelta && !shell.bubble.textContent.trim()) {
        fillAssistant(shell, "服务没有返回内容，请稍后再试。", true);
      }
    } catch (err) {
      fillAssistant(shell, `请求失败：${err.message || err}`, true);
    } finally {
      state.sending = false;
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  // ---------------- 输入框 ----------------

  function resizeInput() {
    inputEl.style.height = "auto";
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 160)}px`;
  }

  inputEl.addEventListener("input", resizeInput);
  inputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      send();
    }
  });
  sendBtn.addEventListener("click", send);

  clearBtn.addEventListener("click", async () => {
    if (state.sending) return;
    if (!window.confirm("清空当前对话记录？会删除本会话的全部消息，不可恢复。")) return;
    const target = state.sessionId;
    rememberSession(null);
    state.sessionId = null;
    renderWelcome();
    if (target) {
      try {
        await api(`/api/sessions/${target}`, { method: "DELETE" });
      } catch (err) {
        console.warn(err);
      }
    }
    inputEl.focus();
  });

  // ---------------- 主题 ----------------

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    themeBtn.textContent = theme === "dark" ? "☀ 浅色" : "☾ 深色";
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", theme === "dark" ? "#0e1117" : "#1d4ed8");
    try {
      localStorage.setItem("campus-qa-theme", theme);
    } catch (err) {
      /* 隐私模式下忽略 */
    }
  }

  const savedTheme = (() => {
    try {
      return localStorage.getItem("campus-qa-theme");
    } catch (err) {
      return null;
    }
  })();
  const prefersDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  applyTheme(savedTheme || (prefersDark ? "dark" : "light"));

  themeBtn.addEventListener("click", () => {
    const current = document.documentElement.dataset.theme;
    applyTheme(current === "dark" ? "light" : "dark");
  });

  // ---------------- 健康检查 ----------------

  async function refreshHealth() {
    try {
      const data = await api("/health");
      healthDot.className = `dot ${data.status === "ok" ? "ok" : "bad"}`;
      const lifeCount = data.life && data.life.doc_count;
      healthText.textContent =
        data.status === "ok"
          ? `服务正常 · 生活库 ${lifeCount ?? 0} 条`
          : "服务降级：生活库未就绪";
    } catch (err) {
      healthDot.className = "dot bad";
      healthText.textContent = "无法连接服务";
    }
  }

  // ---------------- 启动 ----------------

  renderWelcome();
  refreshHealth();
  window.setInterval(refreshHealth, 60000);
  inputEl.focus();

  ensureSession().then((messages) => {
    if (!state.sending) renderHistory(messages);
  });
})();

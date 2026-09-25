// llmrouter site: scroll progress, active nav, install tabs, copy buttons,
// scroll reveal. No dependencies; everything degrades to a readable page.
(() => {
  // ---- reading progress ----------------------------------------------------
  const bar = document.createElement("div");
  bar.id = "scroll-progress";
  document.body.appendChild(bar);
  const setBar = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.width = (max > 0 ? (window.scrollY / max) * 100 : 0) + "%";
  };
  setBar();
  window.addEventListener("scroll", setBar, { passive: true });

  // ---- active nav link ------------------------------------------------------
  const navLinks = [...document.querySelectorAll('.nav nav a[href^="#"]')];
  if ("IntersectionObserver" in window && navLinks.length > 0) {
    const byId = new Map(
      navLinks.map((a) => [a.getAttribute("href").slice(1), a]),
    );
    const spy = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          const link = byId.get(e.target.id);
          if (link) link.classList.toggle("active", e.isIntersecting);
        }
      },
      { rootMargin: "-40% 0px -55% 0px" },
    );
    for (const id of byId.keys()) {
      const sec = document.getElementById(id);
      if (sec) spy.observe(sec);
    }
  }

  // ---- install tabs -------------------------------------------------------
  const tabs = document.querySelectorAll('[role="tab"]');
  for (const tab of tabs) {
    tab.addEventListener("click", () => {
      for (const t of tabs) {
        const selected = t === tab;
        t.setAttribute("aria-selected", String(selected));
        const panel = document.getElementById(t.getAttribute("aria-controls"));
        if (panel) panel.hidden = !selected;
      }
    });
    tab.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const list = [...tabs];
      const i = list.indexOf(tab);
      const next = list[(i + (e.key === "ArrowRight" ? 1 : list.length - 1)) % list.length];
      next.focus();
      next.click();
    });
  }

  // ---- copy buttons -------------------------------------------------------
  for (const btn of document.querySelectorAll(".copy[data-copy]")) {
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(btn.dataset.copy);
      } catch {
        // Clipboard can be denied; the command is selectable text regardless.
      }
      btn.classList.add("done");
      const label = btn.textContent;
      btn.textContent = "copied";
      setTimeout(() => {
        btn.classList.remove("done");
        btn.textContent = label;
      }, 1200);
    });
  }

  // ---- live demo (the real engine, bundled from npm/llmrouter) -------------
  const out = document.getElementById("demo-out");
  const routeBtn = document.getElementById("demo-route");
  const field = (id) => document.getElementById(id);

  const esc = (s) =>
    String(s).replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function renderDecision(r) {
    const maxScore = Math.max(...r.candidates.map((c) => Math.abs(c.score)), 1e-9);
    const rows = r.candidates
      .map((c) => {
        const barW = Math.max(4, Math.round((Math.abs(c.score) / maxScore) * 100));
        return `<tr class="${c.modelId === r.modelId && c.backendId === r.backendId ? "picked" : ""}">` +
          `<td>${esc(c.modelId)}<span class="bar" aria-hidden="true"><span style="width:${barW}%"></span></span></td>` +
          `<td>${esc(c.backendId)}</td>` +
          `<td class="num">${c.score >= 0 ? "+" : ""}${c.score.toFixed(4)}</td>` +
          `<td class="num">$${c.costUsd.toFixed(5)}</td>` +
          `<td class="num ${c.affinity > 0 ? "warm" : "dim"}">${c.affinity > 0 ? c.affinity.toFixed(2) : "&ndash;"}</td>` +
          `<td class="num dim">$${c.switchCostUsd.toFixed(5)}</td></tr>`;
      })
      .join("");

    const ttl = Object.entries(r.ttl)
      .map(([k, v]) => `<span class="ttl-chip ${v === "1h" ? "ttl-1h" : ""}">${esc(k)}&middot;${esc(v)}</span>`)
      .join(" ");

    const flags = [];
    if (r.reusedSession) flags.push("sticky");
    if (r.switched) flags.push("switched");
    if (r.stranded) flags.push("stranded");

    out.innerHTML =
      `<p class="demo-chosen">${esc(r.modelId)} <span class="dim">@ ${esc(r.backendId)}</span></p>` +
      `<p class="demo-meta">workload <strong>${esc(r.workload)}</strong>` +
      ` <span class="dim">via ${esc(r.source)} (conf ${r.confidence.toFixed(2)})</span>` +
      (flags.length ? ` &middot; <strong>${esc(flags.join(", "))}</strong>` : "") +
      ` &middot; quality <strong>${r.quality.toFixed(2)}</strong></p>` +
      `<p class="demo-meta ttl-line">${ttl}</p>` +
      `<table class="demo-table"><thead><tr>` +
      `<th scope="col">candidate</th><th scope="col">backend</th>` +
      `<th scope="col" class="num">score</th><th scope="col" class="num">est $/turn</th>` +
      `<th scope="col" class="num">affinity</th><th scope="col" class="num">switch $</th>` +
      `</tr></thead><tbody>${rows}</tbody></table>` +
      `<ul class="demo-reasons">${r.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;
  }

  function routeNow() {
    if (!out || !routeBtn) return;
    if (!window.LLMRouterDemo) {
      out.innerHTML =
        '<p class="dim">demo bundle not loaded &mdash; build it with ' +
        "<code>npm run build:demo</code> in npm/llmrouter</p>";
      return;
    }
    if (!routeNow.engine) routeNow.engine = window.LLMRouterDemo.createDemoEngine();
    try {
      const r = routeNow.engine.route({
        prompt: field("demo-prompt").value,
        system: field("demo-system").value,
        workload: field("demo-workload").value === "auto"
          ? null
          : field("demo-workload").value,
        sessionId: field("demo-session").value || null,
        expectedGapS: Number(field("demo-gap").value) || 0,
        modelPool: field("demo-pool").value,
      });
      renderDecision(r);
    } catch (err) {
      out.innerHTML = `<p class="dim">routing failed: ${esc(err.message)}</p>`;
    }
  }

  if (routeBtn) {
    routeBtn.addEventListener("click", routeNow);
    routeNow(); // first paint: a real decision, not a placeholder
  }

  // ---- scroll reveal (skipped for reduced motion) --------------------------
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const targets = document.querySelectorAll(
    ".card, .cli-item, .qs-col, .proof-item, .table-wrap, .callout",
  );
  if (reduced || !("IntersectionObserver" in window)) return;
  for (const el of targets) el.classList.add("reveal");
  let stagger = 0;
  const io = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      const el = entry.target;
      el.style.transitionDelay = `${Math.min(stagger * 60, 240)}ms`;
      stagger = (stagger + 1) % 5;
      el.classList.add("in");
      io.unobserve(el);
    }
  }, { threshold: 0.12 });
  targets.forEach((el) => io.observe(el));
})();

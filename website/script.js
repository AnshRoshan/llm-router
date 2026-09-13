// llmrouter site: install tabs, copy buttons, scroll reveal.
// No dependencies; everything degrades to a fully readable static page.
(() => {
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

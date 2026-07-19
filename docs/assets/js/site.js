document.addEventListener("DOMContentLoaded", () => {
  const header = document.querySelector(".md-header");
  const updateHeader = () => header?.classList.toggle("is-scrolled", window.scrollY > 8);
  updateHeader();
  window.addEventListener("scroll", updateHeader, { passive: true });

  document.querySelectorAll(".pygx-copy-command").forEach((button) => {
    button.addEventListener("click", async () => {
      const value = button.dataset.command || "";
      if (!value) return;
      await navigator.clipboard.writeText(value);
      const original = button.textContent;
      button.textContent = "Copied";
      setTimeout(() => { button.textContent = original; }, 1200);
    });
  });
});

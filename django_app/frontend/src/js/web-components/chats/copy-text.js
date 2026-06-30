// @ts-check

const COPY_ICON = `
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <g clip-path="url(#clip0_690_405)">
  <path d="M21 7.5V21H7.5V7.5H21ZM21 6H7.5C7.10218 6 6.72064 6.15804 6.43934 6.43934C6.15804 6.72064 6 7.10218 6 7.5V21C6 21.3978 6.15804 21.7794 6.43934 22.0607C6.72064 22.342 7.10218 22.5 7.5 22.5H21C21.3978 22.5 21.7794 22.342 22.0607 22.0607C22.342 21.7794 22.5 21.3978 22.5 21V7.5C22.5 7.10218 22.342 6.72064 22.0607 6.43934C21.7794 6.15804 21.3978 6 21 6Z"/>
  <path d="M3 13.5H1.5V3C1.5 2.60218 1.65804 2.22064 1.93934 1.93934C2.22064 1.65804 2.60218 1.5 3 1.5H13.5V3H3V13.5Z"/>
  </g>
  <defs>
  <clipPath id="clip0_690_405">
  <rect width="24" height="24" fill="white"/>
  </clipPath>
  </defs>
  </svg>`;

const CHECK_ICON = `
  <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24"><path d="M0 0h24v24H0z" fill="none"/><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>`;

class CopyText extends HTMLElement {
  connectedCallback() {
    const messageId = this.dataset.id;
    this.innerHTML = `
      <button class="ids-copy-button focusable" type="button">
        <span class="ids-copy-icon">${COPY_ICON}</span>
        <span class="ids-copy-label">Copy</span>
        <span class="govuk-visually-hidden"> message content</span>
      </button>
      <span class="ids-copy-status govuk-visually-hidden" role="status" aria-live="polite"></span>
    `;

    this.querySelector("button")?.addEventListener("click", () => {
      const textEl = /** @type {HTMLElement} */ (
        document.querySelector(`#chat-message-${messageId} markdown-converter`));
      this.#copyToClipboard(textEl?.innerHTML, textEl?.innerText);

      const label = this.querySelector(".ids-copy-label");
      const icon = this.querySelector(".ids-copy-icon");
      const status = this.querySelector(".ids-copy-status");
      if (label) label.textContent = "Copied";
      if (icon) icon.innerHTML = CHECK_ICON;
      if (status) status.textContent = "Copied message content to clipboard";

      clearTimeout(this.timer);
      this.timer = setTimeout(() => {
        if (label) label.textContent = "Copy";
        if (icon) icon.innerHTML = COPY_ICON;
        if (status) status.textContent = "";
      }, 2000);
    });
  }

  /**
   * @param {string} html
   * @param {string} text
   */
  #copyToClipboard = (html, text) => {
    const listener = (evt) => {
      evt.clipboardData.setData("text/html", html);
      evt.clipboardData.setData("text/plain", text);
      evt.preventDefault();
    };
    document.addEventListener("copy", listener);
    document.execCommand("copy");
    document.removeEventListener("copy", listener);
  };
}
customElements.define("copy-text", CopyText);

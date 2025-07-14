// @ts-check

export class MessageInput extends HTMLElement {
  constructor() {
    super();
    this.textarea = this.querySelector(".message-input");
  }

  connectedCallback() {
    if (!this.textarea) {
      return;
    }

    // Submit form on enter-key press (providing shift isn't being pressed)
    this.textarea.addEventListener("keypress", (evt) => {
      if (evt.key === "Enter" && !evt.shiftKey && this.textarea) {
        evt.preventDefault();
        if (this.textarea?.textContent?.trim()) {
          this.closest("form")?.requestSubmit();
        }
      }
    });

    // expand textarea as user adds lines
    this.textarea.addEventListener("input", () => {
      this.#adjustHeight();
    });
  }

  #adjustHeight = () => {
    if (!this.textarea) {
      return;
    }
    this.textarea.style.height = "auto";
    this.textarea.style.height = `${this.textarea.scrollHeight || this.textarea.offsetHeight}px`;
  };

  /**
   * Returns the current message
   * @returns string
   */
  getValue = () => {
    return this.textarea?.textContent?.trim() || "";
  };

  /**
   * Clears the message and resets to starting height
   */
  reset = () => {
    if (!this.textarea) {
      return;
    }
    this.textarea.textContent = "";
    this.#adjustHeight();
  };
}
customElements.define("message-input", MessageInput);

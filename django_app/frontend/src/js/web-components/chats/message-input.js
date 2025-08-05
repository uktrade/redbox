// @ts-check

import { hideElement } from "../../utils";

export class MessageInput extends HTMLElement {
  constructor() {
    super();
    this.textarea = /** @type {HTMLDivElement} */ (
      this.querySelector(".message-input")
    );
  }

  connectedCallback() {
    if (!this.textarea) return;

    // Submit form on enter-key press (providing shift isn't being pressed)
    this.textarea.addEventListener("keypress", (evt) => {
      if (evt.key === "Enter" && !evt.shiftKey && this.textarea) {
        this.#hideWarnings();
        evt.preventDefault();
        if (this.textarea?.textContent?.trim()) {
          this.closest("form")?.requestSubmit();
        }
      }
    });
  }


  /**
   * Returns the current message
   * @returns string
   */
  getValue = () => {
    return this.textarea?.textContent?.trim() || "";
  };


  /**
   * Clears the message
   */
  reset = () => {
    if (this.textarea) this.textarea.textContent = "";
  };


  /**
   * Hides the warning messages displayed under the textarea
   */
  #hideWarnings = () => {
    const chatWarnings = /** @type {HTMLDivElement} */ (
      document.querySelector(".chat-warnings")
    );
    if (chatWarnings) hideElement(chatWarnings);
  };
}
customElements.define("message-input", MessageInput);

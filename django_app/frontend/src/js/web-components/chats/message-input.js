// @ts-check

import { pollFileStatus, updateYourDocuments } from "../../services";
import { hideElement } from "../../utils";

export class MessageInput extends HTMLElement {
  constructor() {
    super();
  }

  connectedCallback() {
    this.#bindEvents();
  }


  #sendMessage() {
    if (this.textarea?.textContent?.trim()) {
      this.#hideWarnings();
      this.closest("form")?.requestSubmit();
    }
  }


  #bindEvents(textarea = this.textarea) {
    if (!this.textarea) return;

    this.sendButton?.addEventListener("click", (evt) => {
      evt.preventDefault();
      this.#sendMessage();
    });

    // Submit form on enter-key press (providing shift isn't being pressed)
    textarea.addEventListener("keypress", (evt) => {
      if (evt.key === "Enter" && !evt.shiftKey && this.textarea) {
        evt.preventDefault();
        this.#sendMessage();
      }
    });
  }


  get sendButton() {
    return /** @type {HTMLButtonElement} */ (document.querySelector(".rb-send-button"));
  }


  get textarea() {
    return /** @type {HTMLDivElement} */ (this.querySelector(".message-input"));
  }


  /**
   * Returns the current message, without any uploaded files
   * @returns string
   */
  getValue = () => {
    const clone = this.textarea;
    clone?.querySelector(".uploaded-files-wrapper")?.remove();
    return clone?.textContent?.trim() || "";
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

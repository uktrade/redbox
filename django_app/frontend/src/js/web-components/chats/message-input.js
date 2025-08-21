// @ts-check

import { hideElement } from "../../utils";
import { SendMessage } from "./send-message";
import { SendMessageWithDictation } from "./send-message-with-dictation";

export class MessageInput extends HTMLElement {
  constructor() {
    super();
    this.submitDisabled = false;
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
        if (!this.submitDisabled) this.#sendMessage();
      }
    });
  }


  get sendMessageWithDictation() {
    if (!this._sendMessageWithDictation || !document.body.contains(this._sendMessageWithDictation)) {
      this._sendMessageWithDictation = /** @type {SendMessageWithDictation} */ (
        document.querySelector("send-message-with-dictation")
      );
    }
    return this._sendMessageWithDictation;
  }


  get sendMessage() {
    if (!this._sendMessage || !document.body.contains(this._sendMessage)) {
      this._sendMessage = /** @type {SendMessage} */ (
        document.querySelector("send-message")
      );
    }
    return this._sendMessage;
  }


  get sendButton() {
    return this.sendMessage?.buttonSend || this.sendMessageWithDictation?.buttonSend;
  }


  get dictateButton() {
    return this.sendMessageWithDictation?.buttonRecord;
  }


  get textarea() {
    return /** @type {HTMLDivElement} */ (this.querySelector("#message"));
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
   * Disables submission
   */
  disableSubmit = () => {
    this.submitDisabled = true;
    if (this.sendButton) this.sendButton.disabled = true;
    if (this.dictateButton) this.dictateButton.disabled = true;
  };


  /**
   * Enables submission
   */
  enableSubmit = () => {
    this.submitDisabled = false;
    if (this.sendButton) this.sendButton.disabled = false;
    if (this.dictateButton) this.dictateButton.disabled = false;
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

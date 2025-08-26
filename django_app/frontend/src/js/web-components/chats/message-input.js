// @ts-check

import { UploadedFiles } from "../../../redbox_design_system/rbds/components";
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
    if (this.getValue()) {
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


    textarea.addEventListener("keydown", (evt) => {
      if (!this.textarea) return;

      // Submit form on enter-key press (providing shift isn't being pressed)
      if (evt.key === "Enter" && !evt.shiftKey) {
        evt.preventDefault();
        if (!this.submitDisabled) this.#sendMessage();
      }

      // Prevent deletion of uploaded-files component if present
      if (evt.key === "Backspace") {
        const textContent = this.getValue(false);
        if (textContent === "") evt.preventDefault();
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
   * Disables submission (Can be removed/refactored if send-message and send-message-with-dictation get merged at some point)
   */
  disableSubmit = () => {
    this.submitDisabled = true;
    if (this.sendButton) this.sendButton.disabled = true;
    if (this.dictateButton) this.dictateButton.disabled = true;
  };


  /**
   * Enables submission (Can be removed/refactored if send-message and send-message-with-dictation get merged at some point)
   */
  enableSubmit = () => {
    this.submitDisabled = false;
    if (this.sendButton) this.sendButton.disabled = false;
    if (this.dictateButton) this.dictateButton.disabled = false;
  };


  /**
   * Returns the current message, without any uploaded files
   * @returns string
   */
  getValue = (trim=true) => {
    const clone = /** @type {HTMLElement} */ (this.textarea.cloneNode(true));
    clone.querySelector("uploaded-files")?.remove();
    if (trim) return clone?.textContent?.trim() || "";
    return clone?.textContent || "";
  };


  /**
   * Clears the message
   */
  reset = () => {
    if (!this.textarea) return;
    let hasUploadedFiles = false;
    for (const node of Array.from(this.textarea.childNodes)) {
      switch(node.nodeType) {
        case Node.ELEMENT_NODE:
          if (node instanceof UploadedFiles) hasUploadedFiles = true;
          if (!(node instanceof UploadedFiles)) this.textarea.removeChild(node);
          break;
        default:
          this.textarea.removeChild(node);
      }
    }
    if (hasUploadedFiles) this.textarea.appendChild(document.createElement("br"));
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

// @ts-check

import { hideElement, showElement } from "../../utils";

class SendMessage extends HTMLElement {

  connectedCallback() {
    this.buttonSend = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(2)")
    );
    this.buttonStop = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(3)")
    );

    hideElement(this.buttonStop);
      this.buttonStop.addEventListener("click", () => {
        const stopStreamingEvent = new CustomEvent("stop-streaming");
        document.dispatchEvent(stopStreamingEvent);
      });

      document.addEventListener("chat-response-start", () => {
        if (!this.buttonSend || !this.buttonStop) {
          return;
        }
        hideElement(this.buttonSend);
        showElement(this.buttonStop);
      });

      document.addEventListener("chat-response-end", this.#showSendButton);
      document.addEventListener("stop-streaming", this.#showSendButton);
    }

    #showSendButton = () => {
      if (!this.buttonSend || !this.buttonStop) {
        return;
      }
      showElement(this.buttonSend);
      hideElement(this.buttonStop);
    };
  }
  customElements.define("send-message", SendMessage);

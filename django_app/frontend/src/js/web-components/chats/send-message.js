// @ts-check

import { hideElement, showElement } from "../../utils";

export class SendMessage extends HTMLElement {

  connectedCallback() {
    this.buttonSend = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(1)")
    );
    this.buttonStop = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(2)")
    );

    hideElement(this.buttonStop);
      this.buttonStop.addEventListener("click", () => {
        const stopStreamingEvent = new CustomEvent("stop-streaming");
        document.dispatchEvent(stopStreamingEvent);
      });

      document.addEventListener("chat-response-start", () => {
        hideElement(this.buttonSend);
        showElement(this.buttonStop);
      });

      document.addEventListener("chat-response-end", () => {
        this.showSendButton();
      });

      document.addEventListener("stop-streaming", this.showSendButton);
    }


    /**
     * Show Send button and hide stop send button
     */
    showSendButton = () => {
      showElement(this.buttonSend);
      hideElement(this.buttonStop);
    };


    /**
     * Hide Send button and show stop send button
     */
    hideSendButton() {
      hideElement(this.buttonSend);
      showElement(this.buttonStop);
    }
  }
  customElements.define("send-message", SendMessage);

// @ts-check

import { emitEvent, Events, listenEvent } from "../../../interaction_design_system/ids/events";
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

    this.buttonStop.addEventListener("click", () => emitEvent(Events.STOP_STREAMING));

    listenEvent(Events.CHAT_RESPONSE_START, () => {
      hideElement(this.buttonSend);
      showElement(this.buttonStop);
    });

    listenEvent(Events.CHAT_RESPONSE_END, () => {
      this.showSendButton();
    });

    listenEvent(Events.STOP_STREAMING, this.showSendButton);
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

// @ts-check

import { hideElement } from "../../utils";

class CannedPrompts extends HTMLElement {
  connectedCallback() {
    document.addEventListener("start-streaming", () => {
      hideElement(this);
    });
  }
}

customElements.define("canned-prompts", CannedPrompts);

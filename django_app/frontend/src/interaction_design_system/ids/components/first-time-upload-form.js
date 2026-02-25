// @ts-check

import { hideElement } from "../../../js/utils";

class FirstTimeUploadForm extends HTMLElement {
  connectedCallback() {
    document.addEventListener("start-streaming", () => {
      hideElement(this);
    });
  }
}

customElements.define("ids-first-time-upload-form", FirstTimeUploadForm);

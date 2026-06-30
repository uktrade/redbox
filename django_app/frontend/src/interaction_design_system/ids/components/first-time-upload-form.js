// @ts-check

import { hideElement } from "../../../js/utils";
import { Events, listenEvent } from "../events";

class FirstTimeUploadForm extends HTMLElement {
  connectedCallback() {
    listenEvent(Events.START_STREAMING, () => hideElement(this));
  }
}

customElements.define("ids-first-time-upload-form", FirstTimeUploadForm);

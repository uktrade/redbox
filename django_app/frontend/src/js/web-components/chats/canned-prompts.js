// @ts-check

import { Events, listenEvent } from "../../../interaction_design_system/ids/events";
import { hideElement } from "../../utils";

class CannedPrompts extends HTMLElement {
  connectedCallback() {
    listenEvent(Events.START_STREAMING, () => hideElement(this));
  }
}

customElements.define("canned-prompts", CannedPrompts);

// @ts-check

import { emitEvent, Events } from "../../../interaction_design_system/ids/events";
import { FileStatusTypes } from "./file-status-types";

class FileStatus extends HTMLElement {
  FILE_STATUS_ENDPOINT = "/file-status";
  CHECK_INTERVAL_MS = 6000;

  connectedCallback() {
    if (!this.dataset.id) return;

    this.checkStatus();
  }

  async checkStatus() {
    const response = await fetch(
      `${this.FILE_STATUS_ENDPOINT}?id=${this.dataset.id}`
    );
    const responseObj = await response.json();
    this.textContent = responseObj.status;
    this.dataset.status = responseObj.status.toLowerCase();

    if (responseObj.status.toLowerCase() === FileStatusTypes.COMPLETE) {
      emitEvent(Events.FILE_STATUS_COMPLETE, {fileStatus:this})
    } else {
      window.setTimeout(() => this.checkStatus(), this.CHECK_INTERVAL_MS);
    }
  }
}
customElements.define("file-status", FileStatus);

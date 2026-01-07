// @ts-check

import "../../../redbox_design_system/rbds/components/loading-message.js";

/** So completed docs can be added to this list */
class UploadButton extends HTMLElement {
  connectedCallback() {
    this.closest("form")?.addEventListener("submit", () => {
      this.querySelector("button")?.remove();
      let el = document.createElement("loading-message");
      el.dataset.message = "Uploading";
      this.appendChild(el);
      this.setAttribute("tabindex", "-1");
      this.focus();
    });
  }
}
customElements.define("upload-button", UploadButton);

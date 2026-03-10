// @ts-check

export class LoadingMessage extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `
      <span class="ids-loading-text govuk-body-s" aria-label="${
        this.dataset.dataAriaLabel || this.dataset.message || "Loading"
      }">
        ${this.dataset.message || "Loading"}
      </span>
      <span class="ids-loading-ellipsis govuk-body-s"></span>
    `;
  }


  /**
   * Returns the loading text element used for response feedback
   * @returns {HTMLSpanElement} Loading Message text element
   */
  get loadingText() {
    if (!this._loadingText) {
      this._loadingText = /** @type {HTMLSpanElement} */ (
        this.querySelector(".ids-loading-text")
      );
    }
    return this._loadingText;
  }

}
customElements.define("ids-loading-message", LoadingMessage);

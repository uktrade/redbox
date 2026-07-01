// @ts-check

import { visuallyHideElement } from "../utils";

export class StreamedContent extends HTMLElement {
  connectedCallback() {
    this.initialiseLayers();
    this.initialiseAccessibility();
  }


  /**
   * Create/find required layers.
   */
  initialiseLayers() {
    let srLayer = this.querySelector("[data-sr-layer]");
    let visualLayer = this.querySelector("[data-visual-layer]");

    // Fully initialised already
    if (srLayer && visualLayer) {
      this._screenReaderContainer = /** @type {HTMLElement} */ (srLayer);
      this._visualContainer = /** @type {HTMLElement} */ (visualLayer);
      return;
    }

    const existingHtml = this.innerHTML;

    this.clear();

    this._screenReaderContainer = /** @type {HTMLDivElement} */ (
      document.createElement("div")
    );
    this._screenReaderContainer.setAttribute("data-sr-layer", "");
    visuallyHideElement(this._screenReaderContainer);
    this._screenReaderContainer.setAttribute("aria-live", "polite");
    this._screenReaderContainer.setAttribute("aria-atomic", "false");

    this._visualContainer = /** @type {HTMLDivElement} */ (
      document.createElement("div")
    );
    this._visualContainer.setAttribute("data-visual-layer", "");
    this._visualContainer.innerHTML = existingHtml;

    this.append(
      this._screenReaderContainer,
      this._visualContainer,
    );
  }


  /**
   * Configure accessibility behaviour based on
   * whether we're in streaming mode.
   */
  initialiseAccessibility() {
    if (this.isStreamingMode) {
      this.visualContainer.setAttribute("aria-hidden", "true");
    } else {
      this.visualContainer.removeAttribute("aria-hidden");
    }
  }


  /**
   * Returns true when the SR layer contains content.
   */
  get isStreamingMode() {
    return Boolean(
      this.screenReaderContainer.textContent?.trim(),
    );
  }


  /**
   * Replace visual content and optionally append
   * screen reader text.
   *
   * @param {string} html
   * @param {string | null} textChunk
   */
  update(html, textChunk = null) {
    this.visualContainer.innerHTML = html;

    if (textChunk) {
      this.appendScreenReaderText(textChunk);
    }
  }


  /**
   * Replace the screen reader content entirely.
   *
   * @param {string} text
   */
  setScreenReaderText(text) {
    if (!this.visualContainer.hasAttribute("aria-hidden")) {
      this.visualContainer.setAttribute("aria-hidden", "true");
    }

    this.screenReaderContainer.textContent = text;
  }


  /**
   * Append screen reader text.
   *
   * @param {string} text
   */
  appendScreenReaderText(text) {
    if (!this.visualContainer.hasAttribute("aria-hidden")) {
      this.visualContainer.setAttribute("aria-hidden", "true");
    }

    this.screenReaderContainer.append(text);
  }


  /**
   * Clear streaming state and return to
   * normal template-rendered accessibility.
   */
  disableStreamingMode() {
    this.visualContainer.removeAttribute("aria-hidden");

    // this.screenReaderContainer.textContent = "";
    this.screenReaderContainer.removeAttribute("aria-live");
    this.screenReaderContainer.setAttribute("aria-hidden", "true");
  }


  /**
   * Remove all child nodes.
   */
  clear() {
    this.replaceChildren();
  }


  /**
   * Update final content and disable streaming mode
   *
   * @param {string} finalHtml final html content
   */
  complete(finalHtml) {
    this.update(finalHtml);
    this.disableStreamingMode();
  }


  /**
   * Screen Reader layer container.
   *
   * @returns {HTMLElement}
   */
  get screenReaderContainer() {
    return /** @type {HTMLElement} */ (this._screenReaderContainer);
  }


  /**
   * Visual layer container.
   *
   * @returns {HTMLElement}
   */
  get visualContainer() {
    return /** @type {HTMLElement} */ (this._visualContainer);
  }
}

customElements.define("streamed-content", StreamedContent);

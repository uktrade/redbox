// @ts-check

export class SourcesList extends HTMLElement {
  constructor() {
    super();
    this.sources = [];
  }
  // citations text sometimes contains quotation marks which breaks the link formed by data-text
  // The Function below helps preserve this information by converting it from an HTML attribute thus avoiding any issues.
  escapeHtmlAttribute(str) {
    try {
      return str
        .replace(/&/g, "&amp;")   // Escape &
        .replace(/"/g, "&quot;")  // Escape "
        .replace(/</g, "&lt;")    // Escape <
        .replace(/>/g, "&gt;");   // Escape >
    } catch (error) {
      console.warn("escapeHtmlAttribute error:", error);
      return str; // Fallback to original string
    }
  }
  /**
   * Adds a source to the current message
   * @param {string} fileName
   * @param {string} url
   * @param {string} matchingText
   */
  add = (fileName, url, matchingText) => {

    // prevent duplicate sources
    if (this.sources.some((source) => source.matchingText === matchingText)) {
      return;
    }

    this.sources.push({
      fileName: fileName,
      url: url,
      matchingText: matchingText,
    });

    let html = `
            <h3 class="iai-chat-bubble__sources-heading govuk-heading-s govuk-!-margin-bottom-1">Sources</h3>
            <div class="rbds-display-flex-from-desktop">
            <ol class="rb-footnote-list govuk-!-margin-bottom-0">
        `;
    this.sources.forEach((source) => {
      if (source.fileName) {
      html += `
                <li class="govuk-!-margin-bottom-0">
                    <a class="iai-chat-bubbles__sources-link govuk-link" href="${
                      source.url
                    }" id="footnote-${this.getAttribute("data-id")}-${
        this.sources.length
      }" data-text="${this.escapeHtmlAttribute(source.matchingText)}">${source.fileName || source.url}</a>
                </li>
            `;
    }
    });
    html += `</div></ol>`;
    this.innerHTML = html;
  };

  }
customElements.define("sources-list", SourcesList);

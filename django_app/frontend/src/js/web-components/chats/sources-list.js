// @ts-check

export class SourcesList extends HTMLElement {
  constructor() {
    super();
    this.sources = [];
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
            <div class="iai-display-flex-from-desktop">
            <ol class="rb-footnote-list govuk-!-margin-bottom-0">
        `;
    this.sources.forEach((source) => {
      html += `
                <li class="govuk-!-margin-bottom-0">
                    <a class="iai-chat-bubbles__sources-link govuk-link" href="${
                      source.url
                    }" id="footnote-${this.getAttribute("data-id")}-${
        this.sources.length
      }" data-text="${source.matchingText}">${source.fileName || source.url}</a>
                </li>
            `;
    });
    html += `</div></ol>`;
    this.innerHTML = html;
  };

  }
customElements.define("sources-list", SourcesList);

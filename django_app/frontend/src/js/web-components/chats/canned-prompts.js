// @ts-check

class CannedPrompts extends HTMLElement {
  connectedCallback() {
    this.securityClassification = this.getAttribute("security-classification");
    this.innerHTML = `
      <h3 class="govuk-heading-m">How can Redbox help you today?</h3>
      <div class="govuk-notification-bannerx govuk-!-margin-bottom-4" role="region" aria-labelledby="govuk-notification-banner-title" data-module="govuk-notification-banner">
  <div class="govuk-notification-banner__headerx">
    <h2 class="govuk-notification-banner__titlex" id="govuk-notification-banner-title">
      Important
    </h2>
  </div>
  <div class="govuk-notification-banner__contentx">
    <p class="govuk-body govuk-!-font-weight-bold">
      Redbox can make mistakes. You must check for accuracy before using the output.
    </p>
  </div>
</div>
<div class="govuk-notification-bannerx govuk-!-margin-bottom-8" role="region" aria-labelledby="govuk-notification-banner-title" data-module="govuk-notification-banner">
  <div class="govuk-notification-banner__headerx">
    <h2 class="govuk-notification-banner__titlex" id="govuk-notification-banner-title">
      Important
    </h2>
  </div>
  <div class="govuk-notification-banner__contentx">
    <p class="govuk-body govuk-!-font-weight-bold">
      You can use up to, and including, ${this.securityClassification} documents.
    </p>
  </div>
</div>
    `;

    let buttons = this.querySelectorAll("button");
    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        this.#prepopulateMessageBox(button.textContent?.trim() || "");
      });
    });
  }

  /**
   * @param {string} prompt
   */
  #prepopulateMessageBox = (prompt) => {
    /** @type HTMLInputElement | null */
    let chatInput = document.querySelector(".iai-chat-input__input");
    if (chatInput) {
      chatInput.innerHTML = prompt;
      chatInput.focus();
      chatInput.selectionStart = chatInput.value.length;
    }
  };
}
customElements.define("canned-prompts", CannedPrompts);

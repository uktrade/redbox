// @ts-check

class CannedPrompts extends HTMLElement {
  connectedCallback() {
    this.securityClassification = this.getAttribute("security-classification");

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
  }
}

customElements.define("canned-prompts", CannedPrompts);

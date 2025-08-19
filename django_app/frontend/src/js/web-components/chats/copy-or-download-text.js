// @ts-check

class CopyOrDownloadText extends HTMLElement {
  connectedCallback() {
    const messageId = this.dataset.id;
    this.innerHTML = `
        <button class="govuk-buttonx govuk-button--secondaryx govuk-!-margin-right-4" type="button" id="copy-button">
          Copy
        </button>
        <button class="govuk-buttonx" type="button" id="download-button">
          Download Draft
        </button>
    `;

    // Copy functionality
    this.querySelector("#copy-button")?.addEventListener("click", () => {
      const textEl = /** @type {HTMLElement} */ (
        document.querySelector(`#chat-message-${messageId} markdown-converter`)
      );
      this.#copyToClipboard(textEl?.innerHTML, textEl?.innerText);
    });

    // Download as docx functionality
    this.querySelector("#download-button")?.addEventListener("click", () => {
      const textEl = /** @type {HTMLElement} */ (
        document.querySelector(`#chat-message-${messageId} markdown-converter`)
      );
      this.#downloadDocx(textEl?.innerHTML, textEl?.innerText, messageId);
    });
  }

  /**
   * @param {string} html
   * @param {string} text
   */
  #copyToClipboard = (html, text) => {
    const listener = (evt) => {
      evt.clipboardData.setData("text/html", html);
      evt.clipboardData.setData("text/plain", text);
      evt.preventDefault();
    };
    document.addEventListener("copy", listener);
    document.execCommand("copy");
    document.removeEventListener("copy", listener);
  };

  /**
   * @param {string} html
   * @param {string} text
   * @param {string} messageId
   */
  #downloadDocx = async (html, text, messageId) => {
    try {
      const response = await fetch('/chats/generate_docx/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': this.#getCookie('csrftoken'),
        },
        body: JSON.stringify({
          message_id: messageId,
          html_content: html,
          text_content: text,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate document');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `draft-${messageId}.docx`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading ', error);
      alert('Failed to download the document. Please try again.');
    }
  };

  /**
   * Get cookie value by name
   * @param {string} name
   * @returns {string | undefined}
   */
  #getCookie = (name) => {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
      const cookies = document.cookie.split(';');
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.substring(0, name.length + 1) === (name + '=')) {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
  };
}

customElements.define("copy-or-download-text", CopyOrDownloadText);
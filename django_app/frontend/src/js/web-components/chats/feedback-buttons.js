export class FeedbackButtons extends HTMLElement {
  connectedCallback() {
    this.collectedData = {
      // 2 for thumbs-up, 1 for thumbs-down
      rating: 0,
      text: "",
      chips: /** @type {string[]}*/ ([]),
    };

    // If the messageID already exists (e.g. for SSR messages), render the feedback HTML immediately
    if (this.dataset.id) {
      this.showFeedback(this.dataset.id);
    }
  }

  /**
   * @param {string} messageId
   */
  showFeedback(messageId) {
    this.dataset.id = messageId;

    this.innerHTML = `
      <div class="feedback__container feedback__container--1" tabindex="-1">
        <h3 class="feedback__heading">Is this response useful?</h3>

        <button class="thumb_feedback-btn thumb_feedback-btn--up" type="button">
          <img src="/static/icons/thumbs-up.svg" alt="Thumbs Up" />
        </button>

        <button class="thumb_feedback-btn thumb_feedback-btn--down" type="button">
          <img src="/static/icons/thumbs-down.svg" alt="Thumbs down" />
        </button>
      </div>
    `;

    // Panel 1 Add event listeners for thumbs-up and thumbs-down buttons
    let thumbsUpButton = this.querySelector(".thumb_feedback-btn--up");
    let thumbsDownButton = this.querySelector(".thumb_feedback-btn--down");

    thumbsUpButton?.addEventListener("click", () => {
      if (!this.collectedData) return;

      if (this.collectedData.rating === 2) {
        // Unset thumbs up if already selected
        this.collectedData.rating = 0;
        this.#resetButtons(thumbsUpButton, thumbsDownButton);
      } else {
        this.collectedData.rating = 2; // Thumbs up
        this.#highlightButton(thumbsUpButton, thumbsDownButton);
      }

      this.#sendFeedback();
    });

    thumbsDownButton?.addEventListener("click", () => {
      if (!this.collectedData) return;

      if (this.collectedData.rating === 1) {
        // Unset thumbs down if already selected
        this.collectedData.rating = 0;
        this.#resetButtons(thumbsUpButton, thumbsDownButton);
      } else {
        this.collectedData.rating = 1; // Thumbs down
        this.#highlightButton(thumbsDownButton, thumbsUpButton);
      }

      this.#sendFeedback();
    });
  }

  #highlightButton(selectedButton, otherButton) {
    selectedButton.style.opacity = 1;
    otherButton.style.opacity = 0.2;
  }

  #resetButtons(button1, button2) {
    button1.style.opacity = 1;
    button2.style.opacity = 0.2;
  }
  /**
   * Posts data back to the server
   */
  async #sendFeedback(retry = 0) {
    const MAX_RETRIES = 10;
    const RETRY_INTERVAL_SECONDS = 10;
    const csrfToken =
      /** @type {HTMLInputElement | null} */ (
        document.querySelector('[name="csrfmiddlewaretoken"]')
      )?.value || "";
    try {
      await fetch(`/ratings/${this.dataset.id}/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": csrfToken,
        },
        body: JSON.stringify(this.collectedData),
      });
    } catch (err) {
      if (retry < MAX_RETRIES) {
        window.setTimeout(() => {
          this.#sendFeedback(retry + 1);
        }, RETRY_INTERVAL_SECONDS * 1000);
      }
    }
  }
}
customElements.define("feedback-buttons", FeedbackButtons);
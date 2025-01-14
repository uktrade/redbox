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
    <div class="feedback__container feedback__container--2" hidden tabindex="-1">
      <fieldset class="feedback__chips-container feedback__negative">
        <legend class="feedback__chips-legend">Select all that apply about the response</legend>
        <div class="feedback__chips-inner-container">
          <!-- Factuality Group -->
          <div class="feedback__chip-group">
            <input class="feedback__chip" type="checkbox" id="chip1-factual-${messageId}" data-testid="Factual" />
            <label class="feedback__chip-label" for="chip1-factual-${messageId}">Factual</label>
            <input class="feedback__chip" type="checkbox" id="chip2-inaccurate-${messageId}" data-testid="Inaccurate" />
            <label class="feedback__chip-label" for="chip2-inaccurate-${messageId}">Inaccurate</label>
          </div>
          <!-- Completeness Group -->
          <div class="feedback__chip-group">
            <input class="feedback__chip" type="checkbox" id="chip3-complete-${messageId}" data-testid="Complete" />
            <label class="feedback__chip-label" for="chip3-complete-${messageId}">Complete</label>
            <input class="feedback__chip" type="checkbox" id="chip4-incomplete-${messageId}" data-testid="Incomplete" />
            <label class="feedback__chip-label" for="chip4-incomplete-${messageId}">Incomplete</label>
          </div>
          <!-- Structure Group -->
          <div class="feedback__chip-group">
            <input class="feedback__chip" type="checkbox" id="chip5-structured-${messageId}" data-testid="Structured" />
            <label class="feedback__chip-label" for="chip5-structured-${messageId}">Followed instructions</label>
            <input class="feedback__chip" type="checkbox" id="chip6-unstructured-${messageId}" data-testid="Unstructured" />
            <label class="feedback__chip-label" for="chip6-unstructured-${messageId}">Not as instructed</label>
          </div>
        </div>
      </fieldset>
    <div class="feedback__text-area">
        <label for="text-${messageId}">Or describe with your own words:</label>
        <textarea class="feedback__text-input" id="text-${messageId}" rows="1"></textarea>
        <button class="feedback__submit-btn" type="button">Submit</button>
    </div>    
    </div>
    <div class="feedback__container feedback__container--thank-you" hidden tabindex="-1">
      <p class="feedback__thank-you-message">Thank you for your feedback!</p>
    </div>
     `;
  
    // Panel 1 Add event listeners for thumbs-up and thumbs-down buttons
    let thumbsUpButton = this.querySelector(".thumb_feedback-btn--up");
    let thumbsDownButton = this.querySelector(".thumb_feedback-btn--down");

    thumbsUpButton?.addEventListener("click", () => {
      if (!this.collectedData) return;

      if (this.collectedData.rating === 1) {
        this.collectedData.rating = 0;
        this.#resetButtons(thumbsUpButton, thumbsDownButton);
      } else {
        this.collectedData.rating = 1;
        this.#highlightButton(thumbsUpButton, thumbsDownButton);
      }

      this.#collectChips();
      this.#sendFeedback();
      this.#showPanel(1);
    });

    thumbsDownButton?.addEventListener("click", () => {
      if (!this.collectedData) return;

      if (this.collectedData.rating === 2) {
        this.collectedData.rating = 0;
        this.#resetButtons(thumbsUpButton, thumbsDownButton);
      } else {
        this.collectedData.rating = 2;
        this.#highlightButton(thumbsDownButton, thumbsUpButton);
      }

      this.#collectChips();
      this.#sendFeedback();
      this.#showPanel(1);
    });

    // Panel 3 - text and chips
    let chipGroups = this.querySelectorAll(".feedback__chip-group");
    chipGroups.forEach((group) => {
      let chips = group.querySelectorAll(".feedback__chip");
      chips.forEach((chip) => {
        chip.addEventListener("change", (e) => {
          chips.forEach((otherChip) => {
            if (otherChip !== e.target) {
              otherChip.checked = false;
            }
          });

          this.#collectChips();
          if (this.collectedData.rating > 0) {
            this.#sendFeedback();
          }
        });
      });
    });

    /** @type {HTMLTextAreaElement | null} */
  // Updated Submit button logic
  const textInput = this.querySelector(`#text-${messageId}`);
  this.querySelector(".feedback__submit-btn")?.addEventListener("click", (evt) => {
    evt.preventDefault();
    if (!this.collectedData) return;

    this.collectedData.text = textInput?.value || "";
    this.#sendFeedback();

    // Hide text area and submit button, show thank-you message
    const textArea = this.querySelector(".feedback__text-area");
    textArea.hidden = true;
    const thankYouPanel = this.querySelector(".feedback__container--thank-you");
    thankYouPanel.removeAttribute("hidden");
    thankYouPanel.focus();
  });
}

  #highlightButton(selectedButton, otherButton) {
    selectedButton.style.opacity = 1;
    otherButton.style.opacity = 0.2;
  }

  #resetButtons(button1, button2) {
    button1.style.opacity = 1;
    button2.style.opacity = 1;
  }

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

  #showPanel(panelIndex) {
    if (!this.collectedData) return;

    /** @type {NodeListOf<HTMLElement>} */
    let panels = this.querySelectorAll(".feedback__container");
    panels.forEach((panel) => panel.setAttribute("hidden", ""));
    panels[panelIndex].removeAttribute("hidden");
    panels[panelIndex].focus();
  }

  #collectChips() {
    this.collectedData.chips = [];
    /** @type {NodeListOf<HTMLInputElement>} */
    let chips = this.querySelectorAll(".feedback__chip");
    chips.forEach((chip) => {
      if (chip.checked) {
        const label = this.querySelector(`[for="${chip.id}"]`);
        if (label) {
          this.collectedData.chips.push(label.textContent?.trim() || "");
        }
      }
    });
  }
}

customElements.define("feedback-buttons", FeedbackButtons);

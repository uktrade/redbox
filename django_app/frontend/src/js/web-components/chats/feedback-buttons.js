export class FeedbackButtons extends HTMLElement {
  connectedCallback() {
    this.collectedData = {
      // 1 for thumbs-up, 2 for thumbs-down
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
    <div class="feedback-container feedback-container--1" tabindex="-1">
      <p class="feedback__heading">Is this response useful?</p>
      <button class="thumb_feedback-btn thumb_feedback-btn--up" type="button">
        <img src="/static/icons/thumbs-up.svg" alt="Thumbs Up" aria-labelledby="feedback-heading" />
      </button>
      <button class="thumb_feedback-btn thumb_feedback-btn--down" type="button">
        <img src="/static/icons/thumbs-down.svg" alt="Thumbs down" aria-labelledby="feedback-heading"/>
      </button>
    </div>
     `;

    // 1 Thumbs up, 2 Thumbs down
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

      this.#sendFeedback();
      this.#showPanel();
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

      this.#sendFeedback();
      this.#showPanel();
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
      })
      this.collectedData.chips = [];
    } catch (err) {
      if (retry < MAX_RETRIES) {
        window.setTimeout(() => {
          this.#sendFeedback(retry + 1);
        }, RETRY_INTERVAL_SECONDS * 1000);
      }
    }
  }

  #showPanel() {
    if (!this.collectedData) return;

    /** @type {NodeListOf<HTMLElement>} */
    let chipPanels = document.querySelector(`.feedback__chips-container-${this.dataset.id}`);
    if (!chipPanels) {
    this.closest(".chat-actions-container").insertAdjacentHTML("afterend",`<div class="govuk-form-group" tabindex="-1">
      <fieldset class="govuk-fieldset feedback__chips-container-${this.dataset.id} feedback__negative">
        <legend class="govuk-fieldset__legend govuk-fieldset__legend--s">Select all that apply about the response</legend>
        <div class="govuk-radios">
          <!-- Factuality Group -->
          <div class="govuk-radios__item">
            <input type="radio" class="govuk-radios__input" id="chip1-factual-${this.dataset.id}" data-testid="Factual" />
            <label class="govuk-label govuk-radios__label" for="chip1-factual-${this.dataset.id}">Factual</label>
          </div>
          <div class="govuk-radios__item">
            <input type="radio" class="govuk-radios__input" id="chip2-inaccurate-${this.dataset.id}" data-testid="Inaccurate" />
            <label class="govuk-label govuk-radios__label" for="chip2-inaccurate-${this.dataset.id}">Inaccurate</label>
          </div>
          </div>
          <!-- Completeness Group -->
          <div class="govuk-radios">
          <div class="govuk-radios__item">
            <input type="radio" class="govuk-radios__input" type="checkbox" id="chip3-complete-${this.dataset.id}" data-testid="Complete" />
            <label class="govuk-label govuk-radios__label" for="chip3-complete-${this.dataset.id}">Complete</label>
          </div>
          <div class="govuk-radios__item">
            <input type="radio" class="govuk-radios__input" type="checkbox" id="chip4-incomplete-${this.dataset.id}" data-testid="Incomplete" />
            <label class="govuk-label govuk-radios__label" for="chip4-incomplete-${this.dataset.id}">Incomplete</label>
          </div>
          </div>
          <!-- Structure Group -->
          <div class="govuk-radios">
          <div class="govuk-radios__item">
            <input type="radio" class="govuk-radios__input" type="checkbox" id="chip5-structured-${this.dataset.id}" data-testid="Structured" />
            <label class="govuk-label govuk-radios__label" for="chip5-structured-${this.dataset.id}">Followed instructions</label>
          </div>
          <div class="govuk-radios__item">
            <input type="radio" class="govuk-radios__input" type="checkbox" id="chip6-unstructured-${this.dataset.id}" data-testid="Unstructured" />
            <label class="govuk-label govuk-radios__label" for="chip6-unstructured-${this.dataset.id}">Not as instructed</label>
          </div>
        </div>
        </div>
      </fieldset>
    <div class="feedback__text-area feedback__text-area-${this.dataset.id}">
        <label for="text-${this.dataset.id}">Or describe with your own words:</label>
        <textarea class="feedback__text-input" id="text-${this.dataset.id}" rows="1"></textarea>
        <button class="feedback__submit-btn" id="submit-button-${this.dataset.id}" type="button">Submit</button>
    </div>
    </div>`)}

    this.#addChipEvents();
    this.#addSubmitEvent();
  }

  #collectChips() {
    let chatController = this.closest("chat-controller")
    this.collectedData.chips = [];
    /** @type {NodeListOf<HTMLInputElement>} */
    let chips = chatController.querySelectorAll(".feedback__chip");
    chips.forEach((chip) => {
      if (chip.checked) {
        const label = chatController.querySelector(`[for="${chip.id}"]`);
        if (label) {
          this.collectedData.chips.push(label.textContent?.trim() || "");
        }
      }
    });
  }


  #addChipEvents() {
    let chatController = this.closest("chat-controller")
    let chipGroups = chatController.querySelectorAll(".feedback__chip-group");
    chipGroups.forEach((group) => {
      let chips = group.querySelectorAll(".feedback__chip");
      chips.forEach((chip) => {
        chip.addEventListener("change", (e) => {
          chips.forEach((otherChip) => {
            if (otherChip !== e.target) {
              otherChip.checked = false;
            }
          });

          if (this.collectedData.rating > 0) {
            this.#sendFeedback();
          }
        });
      });
    });
  }

  #addSubmitEvent() {
    /** @type {HTMLTextAreaElement | null} */
  // Updated Submit button logic
  const textInput = document.querySelector(`#text-${this.dataset.id}`);
  document.querySelector(`#submit-button-${this.dataset.id}`)?.addEventListener("click", (evt) => {
    evt.preventDefault();
    this.#collectChips()
    if (!this.collectedData) return;

    this.collectedData.text = textInput?.value || "";
    this.#sendFeedback();

    // Hide text area and submit button
    let chipPanels = document.querySelector(`.feedback__chips-container-${this.dataset.id}`);
    chipPanels?.parentElement.remove()

  });
  }
}

customElements.define("feedback-buttons", FeedbackButtons);

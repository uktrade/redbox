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
          <legend class="feedback__chips-legend">Select all that apply about the response</h3>
          <div class="feedback__chips-inner-container">
          
            <!-- Factuality Group -->
            <div class="feedback__chip-group">
              <input class="feedback__chip" type="checkbox" id="chip1-factual-${messageId}" name="factuality" value="factual" />
              <label class="feedback__chip-label" for="chip1-factual-${messageId}">Factual</label>
          
              <input class="feedback__chip" type="checkbox" id="chip2-inaccurate-${messageId}" name="factuality" value="inaccurate" />
              <label class="feedback__chip-label" for="chip2-inaccurate-${messageId}">Inaccurate</label>
            </div>
          
            <!-- Completeness Group -->
            <div class="feedback__chip-group">
              <input class="feedback__chip" type="checkbox" id="chip3-complete-${messageId}" name="completeness" value="complete" />
              <label class="feedback__chip-label" for="chip3-complete-${messageId}">Complete</label>
          
              <input class="feedback__chip" type="checkbox" id="chip4-incomplete-${messageId}" name="completeness" value="incomplete" />
              <label class="feedback__chip-label" for="chip4-incomplete-${messageId}">Incomplete</label>
            </div>
          
            <!-- Structure Group -->
            <div class="feedback__chip-group">
              <input class="feedback__chip" type="checkbox" id="chip5-structured-${messageId}" name="structure" value="structured" />
              <label class="feedback__chip-label" for="chip5-structured-${messageId}">Structured</label>
          
              <input class="feedback__chip" type="radio" id="chip6-unstructured-${messageId}" name="structure" value="unstructured" />
              <label class="feedback__chip-label" for="chip6-unstructured-${messageId}">Unstructured</label>
            </div>
          </div>
      </fieldset>
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
      this.#showPanel(1);
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



      // this.preventDefault();
      if (!this.collectedData) {
        return;
      }
      /** @type {NodeListOf<HTMLInputElement>} */
      let chips = this.querySelectorAll(".feedback__chip");
      chips.forEach((chip) => {
        if (chip.checked) {
          const text = this.querySelector(`[for="${chip.id}"]`)?.textContent;
          if (text && this.collectedData) {
            this.collectedData.chips.push(text);
          }
        }
      });
      // this.collectedData.text = textInput?.value || "";
      this.#sendFeedback();
      console.log("#sendFeedback")
      this.#showPanel(1);
      console.log("#sendFeedback")
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
  #showPanel(panelIndex) {
    if (!this.collectedData) {
      return;
    }
    /** @type {NodeListOf<HTMLElement>} */
    let panels = this.querySelectorAll(".feedback__container");
    // panels.forEach((panel) => {
    //   panel.setAttribute("hidden", "");
    // });
    panels[panelIndex].removeAttribute("hidden");
    panels[panelIndex].focus();
    console.log(panels[panelIndex])
 }
}
customElements.define("feedback-buttons", FeedbackButtons);
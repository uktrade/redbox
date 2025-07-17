// @ts-check

export class ChatTitle extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `
        <div class="chat-title__heading-container">
            <div class="chat-title__heading-container-inner">
            ${
              this.dataset.title
                ? `
                <h2 class="chat-title__heading govuk-heading-m">${this.dataset.title}</h2>
            `
                : `
                <h2 class="chat-title__heading govuk-heading-s govuk-visually-hidden">Current chat</h2>
            `
            }
            <button id="chat-title-edit-button" class="edit-button" type="button">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <g id="Formatting/Edit" clip-path="url(#clip0_519_205)">
                    <path id="Vector" d="M22.5 19.5H1.5V21H22.5V19.5Z" fill="black"/>
                    <path id="Vector_2" d="M19.05 6.75C19.65 6.15 19.65 5.25 19.05 4.65L16.35 1.95C15.75 1.35 14.85 1.35 14.25 1.95L3 13.2V18H7.8L19.05 6.75ZM15.3 3L18 5.7L15.75 7.95L13.05 5.25L15.3 3ZM4.5 16.5V13.8L12 6.3L14.7 9L7.2 16.5H4.5Z" fill="black"/>
                    </g>
                    <defs>
                    <clipPath id="clip0_519_205">
                    <rect width="24" height="24" fill="black"/>
                    </clipPath>
                    </defs>
                </svg>
                <span class="govuk-visually-hidden"> chat title</span>
            </button>
            </div>
        </div>
        <div class="chat-title__form-container" hidden>
            <label for="chat-title" class="govuk-visually-hidden">Chat Title</label>
            <input type="text" class="chat-title__input" id="chat-title" maxlength="${
              this.dataset.titleLength
            }" value="${this.dataset.title}" tabindex="-1"/>
        </div>
    `;

    this.headingContainer = this.querySelector(
      ".chat-title__heading-container"
    );
    this.formContainer = this.querySelector(".chat-title__form-container");
    /** @type {HTMLButtonElement | null} */
    this.editButton = this.querySelector("#chat-title-edit-button");
    /** @type {HTMLInputElement | null} */
    this.input = this.querySelector(".chat-title__input");
    this.heading = this.querySelector(".chat-title__heading");

    this.editButton?.addEventListener("click", this.#showForm);
    this.heading?.addEventListener("click", this.#showForm);
    this.input?.addEventListener("keydown", (e) => {
      if (!this.input) {
        return false;
      }
      switch (/** @type {KeyboardEvent} */ (e).key) {
        case "Escape":
          this.input.value = this.dataset.title || "";
          this.#hideForm();
          return true;
        case "Enter":
          e.preventDefault();
          this.#update(true);
          return true;
        default:
          return true;
      }
    });
    this.input?.addEventListener("change", (e) => {
      this.#update(true);
    });
    this.input?.addEventListener("blur", (e) => {
      this.#update(true);
    });

    if (!this.dataset.sessionId) {
      document.addEventListener("chat-response-end", this.#onFirstResponse);
    }

    document.addEventListener("chat-title-change", (evt) => {
      let evtData = /** @type {object} */ (evt).detail;
      if (
        evtData.sender !== "chat-title" &&
        evtData.session_id === this.dataset.sessionId
      ) {
        if (this.input) {
          this.input.value = evtData.title;
        }
        this.#update(false);
      }
    });
  }

  #showForm = () => {
    this.headingContainer?.setAttribute("hidden", "");
    this.formContainer?.removeAttribute("hidden");
    this.input?.focus();
  };

  #hideForm = () => {
    this.headingContainer?.removeAttribute("hidden");
    this.formContainer?.setAttribute("hidden", "");
    this.editButton?.focus();
  };

  #onFirstResponse = (e) => {
    this.dataset.title = e.detail.title;
    this.dataset.sessionId = e.detail.session_id;
    document.removeEventListener("chat-response-end", this.#onFirstResponse);
    if (this.input && this.heading) {
      this.input.value = e.detail.title;
      this.heading.textContent = `${e.detail.title}`;
      this.heading.classList.remove("govuk-visually-hidden");
      window.scrollBy(0, this.heading.getBoundingClientRect().height); // to prevent message jumping when this is made visible
    }
  };

  /**
   * @param {boolean} publishChanges Whether to let other components know about this change
   */
  #update = (publishChanges) => {
    const newTitle = this.input?.value;
    if (!newTitle) {
      return;
    }
    this.#send(newTitle);
    this.dataset.title = newTitle;
    if (this.heading) {
      this.heading.textContent = newTitle || "";
    }
    this.#hideForm();
    if (publishChanges) {
      const chatTitleChangeEvent = new CustomEvent("chat-title-change", {
        detail: {
          title: newTitle,
          session_id: this.dataset.sessionId,
          sender: "chat-title",
        },
      });
      document.dispatchEvent(chatTitleChangeEvent);
    }
  };

  /**
   * @param {string} newTitle
   */
  #send = (newTitle) => {
    const csrfToken =
      /** @type {HTMLInputElement | null} */ (
        document.querySelector('[name="csrfmiddlewaretoken"]')
      )?.value || "";
    fetch(`/chat/${this.dataset.sessionId}/title/`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRFToken": csrfToken },
      body: JSON.stringify({ name: newTitle }),
    });
  };
}

customElements.define("chat-title", ChatTitle);

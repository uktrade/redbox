// @ts-check

class ChatHistory extends HTMLElement {
  connectedCallback() {
    this.dataset.initialised = "true";
    this.#limitVisibleChats();
    this.#addShowMoreButton();
  }

  /**
   * Caps the amount of chats at 7 and the rest will be scrollable
   */
  #limitVisibleChats() {
    const chatGroups = this.querySelectorAll(".chat-group-container");
    chatGroups.forEach((group) => {
      const chats = group.querySelectorAll(".chat-list li");
      chats.forEach((chat, index) => {
        chat.style.display = index < 7 ? "block" : "none";
      });
      group.dataset.visibleCount = 7
    });
  }

  /**
   * Displays a 'Show more' button below the chat list
   */
  #addShowMoreButton() {
    const chatGroups = this.querySelectorAll(".chat-group-container");
    chatGroups.forEach((group) => {
      const chats = group.querySelectorAll(".chat-list li");
      const visibleCount = parseInt(group.dataset.visibleCount, 10) || 5;
      if (chats.length > visibleCount) {
        const showMoreDiv = document.createElement("div");
        showMoreDiv.id = "show-more-div";
        const showMoreLink = document.createElement("a");
        showMoreLink.textContent = "Show more...";
        showMoreLink.id = "show-more-button";
        showMoreLink.classList.add("rb-chat-history__link", "govuk-link--inverse");

        showMoreLink.addEventListener("click", () => {
          this.#showMoreChats(group);
        });

        showMoreDiv.appendChild(showMoreLink);

        group.appendChild(showMoreDiv);
      }
    });
  }

  /**
   * Shows next 7 elements also
   *  @param {HTMLElement} group
   */

  #showMoreChats(group) {
    const chats = group.querySelectorAll(".chat-list li");
    let visibleCount = parseInt(group.dataset.visibleCount, 10) || 5;
    const newVisibleCount = Math.min(visibleCount + 5, chats.length);

    chats.forEach((chat, index) => {
      if (index < newVisibleCount) {
        chat.style.display = "block";
      }
    });

    group.dataset.visibleCount = newVisibleCount;

    // Hide the button if all chats are now visible
    if (newVisibleCount >= chats.length) {
      const button = group.querySelector("#show-more-button");
      if (button) {
        button.style.display = "none";
      }
    }
  }

  /**
   * Creates a "Today" heading, if it doesn't already exist
   */
  #createTodayHeading() {
    // Create "Today" heading if it doesn't already exist
    let todayHeadingExists = false;
    const headings = this.querySelectorAll("h3");
    headings.forEach((heading) => {
      if (heading.textContent === "Today") {
        todayHeadingExists = true;
      }
    });
    if (!todayHeadingExists) {
      let newHeading = /** @type {HTMLTemplateElement} */ (
        this.querySelector("#template-chat_history_heading")
      ).content.querySelector("div");
      let newHeadingText = newHeading?.querySelector("h3");
      if (!newHeading || !newHeadingText) {
        return;
      }
      newHeadingText.textContent = "Today";
      this.prepend(newHeading);
    }
  }

  /**
   * Internal method for adding the list-item to the chat history
   * @param {string} chatId
   * @param {string} title
   * @returns {HTMLLIElement}
   */
  #createItem(chatId, title) {
    const newItem = /** @type {HTMLTemplateElement} */ (
      this.querySelector("#template-chat_history_item")
    ).content
      .querySelector("li")
      ?.cloneNode(true);
    let link = /** @type {HTMLElement} */ (newItem).querySelector("a");
    let chatHistoryItem = /** @type {HTMLElement} */ (newItem).querySelector(
      "chat-history-item"
    );
    if (link) {
      link.textContent = title;
      link.setAttribute("href", `/chats/${chatId}`);
    }
    /** @type {HTMLElement} */ (newItem).dataset.chatid = chatId;
    chatHistoryItem?.setAttribute("data-chatid", chatId);
    return /** @type {HTMLLIElement} */ (newItem);
  }

  /**
   * Adds an item to the chat history
   * @param {string} chatId
   * @param {string} title
   */
  addChat(chatId, title) {
    this.#createTodayHeading();
    let item = this.querySelector(`[data-chatid="${chatId}"]`)?.closest("li");
    if (!item) {
      item = this.#createItem(chatId, title.substring(0, 30));
    }
    this.querySelector("ul")?.prepend(item);
    this.#limitVisibleChats();
  }
}

customElements.define("chat-history", ChatHistory);

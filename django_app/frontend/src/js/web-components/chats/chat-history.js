// @ts-check

class ChatHistory extends HTMLElement {
  connectedCallback() {
    this.dataset.initialised = "true";
  }

  /**
   * Caps the amount of chats at 5 and the rest will be scrollable
   */
  #limitVisibleChats() {
    const chatGroups = this.querySelectorAll(".chat-group-container");
    chatGroups.forEach((group) => {
      const chats = group.querySelectorAll(".chat-list li");
      chats.forEach((chat, index) => {
        chat.style.display = index < 5 ? "block" : "none";
      });
    });
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

// @ts-check

import { addShowMore } from '../show-more';

class ChatHistory extends HTMLElement {
  chatLimit = 5;
  connectedCallback() {
    this.dataset.initialised = "true";
    this.#addShowMoreButton();
  }

  /**
   * Caps the amount of chats at 5 and the rest will be scrollable
   */
  #addShowMoreButton() {
    addShowMore({
      container: this.querySelector(".recent-chats"),
      itemSelector: 'li',
      visibleCount: 5,
    })
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
    let item = this.querySelector(`[data-chatid="${chatId}"]`)?.closest("li");
    if (!item) {
      item = this.#createItem(chatId, title.substring(0, 30));
    }
    this.querySelector("ul")?.prepend(item);
    this.#addShowMoreButton();
  }
}

customElements.define("chat-history", ChatHistory);

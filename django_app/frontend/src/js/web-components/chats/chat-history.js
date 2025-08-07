// @ts-check

import htmx from 'htmx.org';

export class ChatHistory extends HTMLElement {

  connectedCallback() {
    this.dataset.initialised = "true";
  }


  /**
   * DEPRECIATED - Internal method for adding the list-item to the chat history
   * @param {string} chatId
   * @param {string} title
   * @returns {HTMLLIElement}
   */
  #createItem(chatId, title) {
    const newItem = /** @type {HTMLTemplateElement} */ (
      this.querySelector("#template-chat_history_item")
    );
    let chatListItem = /** @type {HTMLLIElement} */ (
      newItem.content.querySelector("li")?.cloneNode(true)
    );
    let link = /** @type {HTMLElement} */ (chatListItem).querySelector("a");
    let chatHistoryItem = /** @type {HTMLElement} */ (chatListItem).querySelector(
      "chat-history-item"
    );
    let editableText = /** @type {HTMLElement} */ (
      chatHistoryItem?.querySelector("editable-text")
    );

    editableText?.setAttribute("post-url", `/chat/${chatId}/title/`);
    editableText?.setAttribute("object-id", chatId);

    if (chatListItem) {
      chatListItem.id = `chat-${chatId}`;
      chatListItem.dataset.chatid = chatId;
      chatListItem.classList.add("selected");
    }

    if (link) {
      link.textContent = title;
      link.setAttribute("href", `/chats/${chatId}`);
    }
    // /** @type {HTMLElement} */ (chatListItem).dataset.chatid = chatId;
    chatHistoryItem?.setAttribute("data-chatid", chatId);

    let editButton = /** @type {HTMLButtonElement} */ (
      chatListItem.querySelector(".edit-button")
    );

    let deleteButton = /** @type {HTMLButtonElement} */ (
      chatListItem.querySelector("#delete-chat-confirm-button")
    );

    let deleteConfirmTitle = /** @type {HTMLSpanElement} */ (
      chatListItem.querySelector("#delete-chat-confirm-title")
    );
    deleteConfirmTitle.innerText = title;
    // TODO: Slight refactor upon completion of delete confirmation
    if (deleteButton) {
      deleteButton.setAttribute("hx-post", `/chats/${chatId}/delete-chat/`);
      deleteButton.setAttribute("hx-vals", `{"active_chat_id": "${chatId}"}`);
      deleteButton.setAttribute("hx-target", `#chat-${chatId}`);
    }
    htmx.process(deleteButton);

    return /** @type {HTMLLIElement} */ (chatListItem);
  }


  /**
   * DEPRECIATED - Adds an item to the chat history
   * @param {string} chatId
   * @param {string} title
   */
  addChat(chatId, title) {
    let item = this.querySelector(`[data-chatid="${chatId}"]`)?.closest("li");
    if (!item) {
      item = this.#createItem(chatId, title.substring(0, 30));
    }
    this.querySelector("ul")?.prepend(item);
  }


  /**
   * Update item position
   * @param {string} chatId
   */
  moveToTop(chatId) {
    let item = this.querySelector(`[data-chatid="${chatId}"]`)?.closest("li");
    if (!item) return;
    this.querySelector("ul")?.prepend(item);
  }
}

customElements.define("chat-history", ChatHistory);

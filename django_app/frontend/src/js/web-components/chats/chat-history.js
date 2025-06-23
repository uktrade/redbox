// @ts-check

import { addShowMore } from "../show-more.js";

class ChatHistory extends HTMLElement {
  chatLimit = 5;
  connectedCallback() {
    this.dataset.initialised = "true";
    // addShowMore({
    //   container: this.querySelector(".recent-chats"),
    //   itemSelector: 'li',
    //   itemDisplay: 'block',
    //   visibleCount: 5,
    // })
    this.#addShowMoreButton();
    // this.#limitVisibleChats();
    // this.#addShowMoreButton();
  }

  /**
   * Caps the amount of chats at 5 and the rest will be scrollable
   */
  #addShowMoreButton() {
    addShowMore({
      container: this.querySelector(".recent-chats"),
      itemSelector: 'li',
      itemDisplay: 'block',
      visibleCount: 5,
    })
  }

  /**
   * Caps the amount of chats at 5 and the rest will be scrollable
   */
  #limitVisibleChats() {

    const chatGroup = document.getElementById("recent-chats");
    if (chatGroup) {
      const chats = chatGroup.querySelectorAll("li");
      chats.forEach((chat, index) => {
        chat.style.display = index < this.chatLimit ? "block" : "none";
      });
      chatGroup.dataset.visibleCount = String(this.chatLimit);
    }
  }

  /**
   * Displays a 'Show more' button below the chat list
   */
  // #addShowMoreButton() {
  //   const chatGroup = document.getElementById("recent-chats");
  //   if (chatGroup) {
  //     const chats = chatGroup.querySelectorAll("li");
  //     let visibleCount = this.chatLimit

  //     if (chats.length > visibleCount) {
  //       const showMoreDiv = document.createElement("div");
  //       showMoreDiv.id = "show-more-div";
  //       const showMoreLink = document.createElement("a");
  //       showMoreLink.textContent = "Show more...";
  //       showMoreLink.id = "show-more-button";
  //       showMoreLink.classList.add("rb-chat-history__link", "govuk-link--inverse");

  //       showMoreLink.addEventListener("click", () => {
  //         this.#showMoreChats(chatGroup);
  //       });

  //       showMoreDiv.appendChild(showMoreLink);
  //       chatGroup.appendChild(showMoreDiv);
  //     }
  //   }
  // }

  /**
   * Shows next 7 elements also
   *  @param {HTMLElement} group
   */

  // #showMoreChats(group) {
  //   const chats = group.querySelectorAll("li");
  //   if (group) {
  //     chats.forEach((chat) => {
  //       chat.style.display = "block";
  //     });

  //     // Hide the button now that all chats are visible
  //     const button = document.getElementById("show-more-button");;
  //       if (button) {
  //         button.style.display = "none";
  //       }
  //   }
  // }

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
    // this.#createTodayHeading();
    let item = this.querySelector(`[data-chatid="${chatId}"]`)?.closest("li");
    if (!item) {
      item = this.#createItem(chatId, title.substring(0, 30));
    }
    this.querySelector("ul")?.prepend(item);
    this.#addShowMoreButton();
    // this.#limitVisibleChats();
  }
}

customElements.define("chat-history", ChatHistory);

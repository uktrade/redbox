// @ts-check

import { ChatMessage } from "./chat-message";

class ChatController extends HTMLElement {

  connectedCallback() {
    this.#bindEvents();
  }

  #bindEvents = () => {
    const chatsForm = document.querySelector("#chats-form");
    let selectedDocuments = [];

    chatsForm?.addEventListener("submit", (evt) => {
      evt.preventDefault();

      const chatController = /** @type {ChatController} */ (
        document.querySelector("chat-controller")
      );

      const messageContainer = chatController.querySelector(".js-message-container");
      messageContainer?.classList.add("test-update-dom");
      const insertPosition = chatController.querySelector(".js-response-feedback");
      const feedbackButtons = /** @type {HTMLElement | null} */ (
        chatController.querySelector("feedback-buttons")
      );
      const messageInput = /** @type {import("./message-input").MessageInput} */ (
          document.querySelector("message-input")
      );
      const userText = messageInput?.getValue();

      if (!messageInput || !userText) return;

      let userMessage = /** @type {ChatMessage} */ (
        document.createElement("chat-message")
      );
      userMessage.setAttribute("data-text", userText);
      userMessage.setAttribute("data-role", "user");
      messageContainer?.insertBefore(userMessage, insertPosition);

      let documents = [];
      if (selectedDocuments.length) {
        selectedDocuments.forEach(document => { documents.push(document.name)})
      }

      documents.forEach((activity) => {
        userMessage.addFile(activity);
      });

      let aiMessage = /** @type {import("./chat-message").ChatMessage} */ (
        document.createElement("chat-message")
      );
      aiMessage.setAttribute("data-role", "ai");
      messageContainer?.insertBefore(aiMessage, insertPosition);

      const llm =
        /** @type {HTMLInputElement | null}*/ (
          document.querySelector("#llm-selector")
        )?.value || "";

      aiMessage.stream(
        userText,
        selectedDocuments.map(doc => doc.id),
        documents,
        llm,
        chatController.dataset.sessionId,
        chatController.dataset.streamUrl || "",
        chatController
      );
      /** @type {HTMLElement | null} */ (
        aiMessage.querySelector(".govuk-inset-text")
      )?.focus();

      // reset UI
      if (feedbackButtons) feedbackButtons.dataset.status = "";
      messageInput.reset();
    });

    document.body.addEventListener("selected-docs-change", (evt) => {
      selectedDocuments = /** @type{CustomEvent} */ (evt).detail;
    });
  }
}
customElements.define("chat-controller", ChatController);

// @ts-check

import { getActiveToolId } from "../../utils";
import { ChatMessage } from "./chat-message";

class ChatController extends HTMLElement {

  connectedCallback() {
    this.#bindEvents();
  }

  #bindEvents = () => {
    const chatsForm = document.querySelector("#chats-form");
    let selectedDocuments = [];
    const selectedTool = getActiveToolId();

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
          document.querySelector("rbds-message-input")
      );
      const userText = messageInput?.getValue();
      const hasContent = Boolean(userText || messageInput?.hasUploadedFiles());

      if (!messageInput || !hasContent) return;

      let userMessage = /** @type {ChatMessage} */ (
        document.createElement("rbds-chat-message")
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
        document.createElement("rbds-chat-message")
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
        chatController,
        selectedTool
      );
      /** @type {HTMLElement | null} */ (
        aiMessage.querySelector(".govuk-inset-text")
      )?.focus();

      // reset UI
      if (feedbackButtons) feedbackButtons.dataset.status = "";
      messageInput.reset(true);
    });

    document.body.addEventListener("selected-docs-change", (evt) => {
      selectedDocuments = /** @type{CustomEvent} */ (evt).detail;
    });
  }
}
customElements.define("chat-controller", ChatController);

// @ts-check

class ChatController extends HTMLElement {
  connectedCallback() {
    const messageForm = this.closest("form");
    const messageContainer = this.querySelector(".js-message-container");
    const insertPosition = this.querySelector(".js-response-feedback");
    const feedbackButtons = /** @type {HTMLElement | null} */ (
      this.querySelector("feedback-buttons")
    );
    let selectedDocuments = [];

    messageForm?.addEventListener("submit", (evt) => {
      evt.preventDefault();
      const messageInput =
        /** @type {import("./message-input").MessageInput} */ (
          document.querySelector("message-input")
        );
      const userText = messageInput?.getValue();
      if (!messageInput || !userText) {
        return;
      }

      let userMessage = /** @type {import("./chat-message").ChatMessage} */ (
        document.createElement("chat-message")
      );
      userMessage.setAttribute("data-text", userText);
      userMessage.setAttribute("data-role", "user");
      messageContainer?.insertBefore(userMessage, insertPosition);

      let activites = [];
      if (selectedDocuments.length) {
        activites.push(`You selected ${selectedDocuments.length} document${
          selectedDocuments.length === 1 ? "" : "s"
        }`);
      }
      activites.push("You sent this prompt");

      // add filename to activity if only one file
      if (selectedDocuments.length === 1) {
        activites[0] += ` (${selectedDocuments[0].name})`;
      }
      activites.forEach((activity) => {
        userMessage.addActivity(activity, "user");
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
        activites,
        llm,
        this.dataset.sessionId,
        this.dataset.streamUrl || "",
        this
      );
      /** @type {HTMLElement | null} */ (
        aiMessage.querySelector(".iai-chat-bubble")
      )?.focus();

      // reset UI
      if (feedbackButtons) {
        feedbackButtons.dataset.status = "";
      }
      messageInput.reset();
      
    });

    document.body.addEventListener("selected-docs-change", (evt) => {
      selectedDocuments = /** @type{CustomEvent} */ (evt).detail;
    });
  }
}
customElements.define("chat-controller", ChatController);

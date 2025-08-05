import "./web-components/chats/chat-controller.js";
import "./web-components/chats/chat-history.js";
import "./web-components/chats/chat-history-item.js";
import "./web-components/chats/chat-message.js";
import "./web-components/chats/chat-title.js";
import "./web-components/chats/copy-text.js";
import "./web-components/chats/document-selector.js";
import "./web-components/chats/feedback-buttons.js";
import "./web-components/markdown-converter.js";
import "./web-components/chats/message-input.js";
import "./web-components/chats/sources-list.js";
import "./web-components/chats/canned-prompts";
import "./web-components/chats/send-message.js";
import "./web-components/chats/send-message-with-dictation.js";
import "./web-components/chats/profile-overlay.js";
import "./web-components/documents/file-status.js";
import "./web-components/chats/profile-overlay.js";
import "./web-components/chats/exit-feedback.js";
import { updateChatWindow, syncUrlWithContent } from "./services";

// RBDS - tbc
import "../stylesheets/components/show-more.js";
import "../stylesheets/components/editable-text.js";
import { getActiveChatId } from "./utils/active-chat.js";


document.addEventListener("chat-response-end", (evt) => {
  const sessionId = /** @type{CustomEvent} */ (evt).detail.session_id;
  const sessionTitle = /** @type{CustomEvent} */ (evt).detail.title;

  // Update URL when a new chat is created
  if (sessionId !== getActiveChatId()) {
    window.history.pushState({}, "", `/chats/${sessionId}`);
  }

  // And add to chat history or move to top of recent chats if chat exists
  document.querySelector("chat-history").addChat(sessionId, sessionTitle);

  // Reload chat window to fix citation references
  updateChatWindow(sessionId);

});

syncUrlWithContent();

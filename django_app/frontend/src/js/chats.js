import "./web-components/chats/chat-controller.js";
import "./web-components/chats/chat-history.js";
import "./web-components/chats/chat-history-item.js";
import "./web-components/chats/chat-message.js";
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
import "./web-components/documents/file-upload.js";

import { updateChatWindow, syncUrlWithContent, updateRecentChatHistory } from "./services";
import { ChatHistory } from "./web-components/chats/chat-history.js";
import { getActiveChatId } from "./utils/active-chat.js";


document.addEventListener("chat-response-end", (evt) => {
  const event = /** @type {CustomEvent} */ (evt);

  const sessionId = event.detail.session_id;
  const isNewChat = event.detail.is_new_chat;

  // If new chat or first message was stopped prematurely
  if (isNewChat || !getActiveChatId()) {
    // Update Recent chats section
    updateRecentChatHistory(sessionId);
  } else {
    // Move current chat to top of list
    /** @type {ChatHistory} */ (document.querySelector("chat-history")).moveToTop(sessionId);
  }

  // Reload chat window to fix citation references
  updateChatWindow(sessionId);
});

syncUrlWithContent();

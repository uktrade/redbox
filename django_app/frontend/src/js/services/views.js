// @ts-check

import htmx from "htmx.org";
import { getActiveChatId } from "../utils";

/**
 * Reloads chat window component
 * @param {string | null} chatId - Active chat ID
*/
export function updateChatWindow(chatId = getActiveChatId()) {;
    const url = chatId ? `/chats/${chatId}/chat-window/` : "/chats/chat-window/";
    htmx.ajax('get', url, {
    target: '#chat-window',
    swap: 'outerHTML',
    })
}


/**
 * Reloads recent chat history component
 * @param {string | null} chatId - Active chat ID
*/
export function updateRecentChatHistory(chatId = getActiveChatId()) {;
    const url = chatId ? `/chats/${chatId}/recent-chats/` : "/chats/recent-chats/";
    htmx.ajax('get', url, {
      target: 'chat-history',
      swap: 'outerHTML',
    })
}

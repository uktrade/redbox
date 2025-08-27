// @ts-check

import htmx from "htmx.org";
import { getActiveChatId } from "../utils";

/**
 * Reloads chat window component
 * @param {string | null} chatId - Active chat ID
*/
export function updateChatWindow(chatId = getActiveChatId()) {;
    const url = chatId ? `/chats/${chatId}/chat-window/` : "/chats/chat-window/";
    return htmx.ajax('get', url, {
    target: '#chat-window',
    swap: 'outerHTML',
    });
}


/**
 * Reloads recent chats side-panel template
 * @param {string | null} chatId - Active chat ID
*/
export function updateRecentChatHistory(chatId = getActiveChatId()) {;
    const url = chatId ? `/chats/${chatId}/recent-chats/` : "/chats/recent-chats/";
    return htmx.ajax('get', url, {
      target: 'chat-history',
      swap: 'outerHTML',
    });
}


/**
 * Reloads Your documents side-panel template
 * @param {string | null} chatId - Active chat ID
*/
export function updateYourDocuments(chatId = getActiveChatId()) {;
    const url = chatId ? `/documents/your-documents/${chatId}/` : "/documents/your-documents/";
    return htmx.ajax('get', url, {
      target: 'document-selector',
      swap: 'outerHTML',
    });
}


/**
 * Fetch the icon svg for a given file extension
 * @param {string | undefined} ext - file extension
 * @returns {Promise<string | undefined>}
 */
export async function loadIcon(ext) {
    if (!ext) return;
    const response = await fetch(`/file-icon/${ext}/`);
    if (!response.ok) throw new Error(`Icon not found for extension: ${ext}`);
    const html = await response.text();
    return html;
}

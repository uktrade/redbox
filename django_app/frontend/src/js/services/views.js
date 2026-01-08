// @ts-check

import htmx from "htmx.org";
import { getActiveChatId, getActiveToolSlug } from "../utils";

/**
 * Reloads chat window component
 * @param {string | null} chatId - Active chat ID
*/
export function updateChatWindow(chatId = getActiveChatId(), slug = getActiveToolSlug()) {;
    const tool_url_fragment = slug ? `/tools/${slug}` : "";
    const chat_url_fragment = chatId ? `/${chatId}` : "";
    const url = `${tool_url_fragment}/chats${chat_url_fragment}/chat-window/`;

    return htmx.ajax('get', url, {
    target: '#chat-window',
    swap: 'outerHTML',
    });
}


/**
 * Reloads recent chats side-panel template
 * @param {string | null} chatId - Active chat ID
*/
export function updateRecentChatHistory(chatId = getActiveChatId(), slug = getActiveToolSlug()) {;
    const tool_url_fragment = slug ? `/tools/${slug}` : "";
    const chat_url_fragment = chatId ? `/${chatId}` : "";
    const url = `${tool_url_fragment}/chats${chat_url_fragment}/recent-chats/`;

    return htmx.ajax('get', url, {
      target: 'chat-history',
      swap: 'outerHTML',
    });
}


/**
 * Reloads Your documents side-panel template
 * @param {string | null} chatId - Active chat ID
*/
export function updateYourDocuments(chatId = getActiveChatId(), slug = getActiveToolSlug()) {;
    const tool_url_fragment = slug ? `/tools/${slug}` : "";
    const chat_url_fragment = chatId ? `/${chatId}` : "";
    const url = `${tool_url_fragment}/documents/your-documents${chat_url_fragment}/`;

    return htmx.ajax('get', url, {
      target: 'document-selector',
      swap: 'outerHTML settle:0ms',
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

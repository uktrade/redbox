// @ts-check

/**
 * Extract the active chat id from url
*/
export function getChatIdFromUrl() {
    const currentUrl = window.location.pathname;
    const uuidRegex = /chats\/([0-9a-fA-F-]{36})\/?/;
    const match = currentUrl.match(uuidRegex);
    const currentUUID = match ? match[1] : null;

    return currentUUID;
}


/**
 * Extract the active chat id from the currently selected chat
*/
export function getSelectedChatId() {
    const selectedChat = /** @type {HTMLElement} */ (
        document.querySelector(".chat-list-item.selected")
    );
    const selectedUUID = selectedChat?.dataset?.chatid ?? null;

    return selectedUUID;
}


/**
 * Get active chat id from selected chat or url
*/
export function getActiveChatId() {
    return getSelectedChatId() || getChatIdFromUrl();
}

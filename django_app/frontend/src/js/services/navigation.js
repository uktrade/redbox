// @ts-check

import htmx from "htmx.org";
import { getChatIdFromUrl, getSelectedChatId } from "../utils";

/**
 * Listens for changes to page content and updates url accordingly
*/
export function syncUrlWithContent() {
    // Note: A mutationObserver on the document body may be better here
    // once a refresh is no longer required to fix citation references
    htmx.onLoad(function (content) {
        const template = /** @type{HTMLElement} */ (content);

        switch (template.id) {

            // Update URL whenever content is updated
            case "recent-chats":
            case "chat-window":
                const selectedUUID = getSelectedChatId();
                const currentUUID = getChatIdFromUrl();

                if (selectedUUID !== currentUUID) {
                    const newUrl = selectedUUID ? `/chats/${selectedUUID}/` : '/chats/';

                    if (currentUUID && !selectedUUID) {
                        // Chat deleted, replace current history entry
                        window.history.replaceState({}, "", newUrl);
                    } else {
                        // Add history entry
                        window.history.pushState({}, "", newUrl);
                    }
                }

        }
    });
}

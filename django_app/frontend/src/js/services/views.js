// @ts-check

import htmx from "htmx.org";
import { getActiveChatId, getActiveToolSlug } from "../utils";


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


/**
 * Reloads page fragments
 * @param {(string|any)[]} fragments - List of page section/element id's to reload
 * @param {string | null} chatId - Active chat ID
 * @param {string | undefined} slug - Tool slug
*/
export function refreshUI(fragments, chatId = getActiveChatId(), slug = getActiveToolSlug()) {
    const params = new URLSearchParams()
    if (chatId) params.set("chat", chatId);
    if (slug) params.set("tool", slug);

    fragments.forEach((/** @type {string | any} */ fragment) =>
        params.append("fragments", fragment)
    );

    return htmx.ajax('get', `/ui/refresh?${params}`, {
      target: 'body',
      swap: 'none settle:0ms',
    });
}

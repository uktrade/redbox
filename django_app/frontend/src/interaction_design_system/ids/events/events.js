// @ts-check

/**
 * Central event registry
 */
export const Events = /** @type {const} */ ({
    /** When the streaming connection is opened **/
    CHAT_RESPONSE_START: "chat-response-start",

    /** When the stream "end" event is sent from the server **/
    CHAT_RESPONSE_END: "chat-response-end",

    /** When a document status changes to "complete" **/
    DOC_COMPLETE: "doc-complete",

    /** When a user selects or deselects a document **/
    SELECTED_DOCS_CHANGE: "selected-docs-change",

    /** When a user submits a message **/
    START_STREAMING: "start-streaming",

    /**
     * When a user presses the stop-streaming button,
     * or an unexpected disconnection has occured
    **/
    STOP_STREAMING: "stop-streaming",

    /** When the chat title is changed by the user **/
    CHAT_TITLE_CHANGE: "chat-title-change",

    /** When a individual file has finished processing **/
    FILE_UPLOAD_PROCESSED: "file-upload-processed",

    /** When all file uploads have finished processing **/
    FILE_UPLOADS_PROCESSED: "file-uploads-processed",

    /** When all file uploads have been removed **/
    FILE_UPLOADS_REMOVED: "file-uploads-removed",

    /** When a document has been selected/deselected in the side panel **/
    DOC_SELECTION_CHANGE: "doc-selection-change",

    /** When the side-panel has been toggled **/
    SIDE_PANEL_TOGGLE: "side-panel-toggle",

    /** Trigger a page scroll to bottom **/
    SCROLL_TO_BOTTOM: "scroll-to-bottom",
})

/**
 * @typedef {{
 *  "chat-response-start": undefined,
 *  "chat-response-end": {title:string, session_id:string, is_new_chat:boolean},
 *  "doc-complete": {fileStatus:HTMLElement},
 *  "selected-docs-change": {id:string, name:string}[],
 *  "start-streaming": undefined,
 *  "stop-streaming": undefined,
 *  "chat-title-change": {title:string, session_id:string, sender:string},
 *  "file-upload-processed": undefined,
 *  "file-uploads-processed": undefined,
 *  "file-uploads-removed": undefined,
 *  "doc-selection-change": {id:string, name:string, checked:boolean},
 *  "side-panel-toggle": undefined,
 * "scroll-to-bottom": {source:HTMLElement, force?:boolean},
 * }} EventMap
 */

/** @type {EventTarget} */
let eventTarget = document;

/**
 * Configure event target
 * @param {EventTarget} target;
 */
export function setEventTarget(target) {
    eventTarget = target;
}

/**
 * Emit event
 * @template {keyof EventMap} T
 * @param {T} name
 * @param {EventMap[T] | undefined} detail
 */
export function emitEvent(name, detail=undefined) {
    eventTarget.dispatchEvent(new CustomEvent(name, { detail }));
}

/**
 * Listen to event
 * @template {keyof EventMap} T
 * @param {T} name
 * @param {(event: CustomEvent<EventMap[T]>) => void} handler
 */
export function listenEvent(name, handler) {
    eventTarget.addEventListener(name, /** @type {EventListener} */(handler));
}

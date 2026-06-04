// @ts-check

/**
 * Central event registry
 */
export const Events = /** @type {const} */ ({
    // Project Events

    /** When the streaming connection is opened **/
    CHAT_RESPONSE_START: "chat-response-start",

    /** When the stream "end" event is sent from the server **/
    CHAT_RESPONSE_END: "chat-response-end",

    /** When the streaming connection errors **/
    CHAT_RESPONSE_ERROR: "chat-response-error",

    /** When a document status changes to "complete" **/
    DOC_COMPLETE: "doc-complete",

    /** When a document status changes to "errored" **/
    DOC_ERROR: "doc-error",

    /** When a user selects or deselects a document **/
    SELECTED_DOCS_CHANGE: "selected-docs-change",

    /** When a user submits a message **/
    START_STREAMING: "start-streaming",

    /**
     * When a user presses the stop-streaming button,
     * or an unexpected disconnection has occured
    **/
    STOP_STREAMING: "stop-streaming",

    /** When a document has been selected/deselected in the side panel **/
    DOC_SELECTION_CHANGE: "doc-selection-change",

    /** When the FileStatus element is complete **/
    FILE_STATUS_COMPLETE: "file-status-complete",

    // IDS Events

    /** When an editable-text element is changed by the user **/
    EDITABLE_TEXT_CHANGE: "editable-text-change",

    /** When an editable-text element is deleted **/
    EDITABLE_TEXT_DELETE: "editable-text-delete",

    /** When a individual file has finished processing **/
    FILE_UPLOAD_PROCESSED: "file-upload-processed",

    /** When all file uploads have finished processing **/
    FILE_UPLOADS_PROCESSED: "file-uploads-processed",

    /** When all file uploads have been removed **/
    FILE_UPLOADS_REMOVED: "file-uploads-removed",

    /** When the side-panel has been toggled **/
    SIDE_PANEL_TOGGLE: "side-panel-toggle",

    /** Trigger a page scroll to bottom **/
    SCROLL_TO_BOTTOM: "scroll-to-bottom",
});

/**
 * @typedef {{
 *  "chat-response-start": undefined,
 *  "chat-response-end": {title:string, session_id:string, is_new_chat:boolean},
 *  "chat-response-error": undefined,
 *  "doc-complete": {id:string, status:string},
 *  "doc-error": {id:string, status:string},
 *  "selected-docs-change": {id:string, name:string}[],
 *  "start-streaming": undefined,
 *  "stop-streaming": undefined,
 *  "editable-text-change": {sender_id:string, object_id?:string, value:string},
 *  "editable-text-delete": {sender_id:string, object_id?:string},
 *  "doc-selection-change": {id:string, name:string, checked:boolean},
 *  "file-status-complete": {fileStatus:HTMLElement},
 *  "file-status-error": {id:string, status:string},
 *  "file-upload-processed": {uploadedFile:HTMLElement},
 *  "file-uploads-processed": {uploadedFiles:HTMLElement},
 *  "file-uploads-removed": {uploadedFiles:HTMLElement},
 *  "side-panel-toggle": {SidePanelToggle:HTMLElement},
 *  "scroll-to-bottom": {source:HTMLElement, force?:boolean},
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

// @ts-check

import { emitEvent, Events, listenEvent } from "../../../interaction_design_system/ids/events";
import { getActiveToolId, sanitizeHtml } from "../../utils";
import { ChatMessage } from "./chat-message";

const STATE = {
    EMPTY: "empty",
    CONVERSATION: "conversation",
}

const STATUS = {
    STREAMING: "streaming",
    COMPLETE: "complete",
    STOPPED: "stopped",
    ERROR: "error",
}

export class ChatController extends HTMLElement {
    constructor() {
        super();

        this.dataset.state = STATE.EMPTY;

        /** @type {SelectedDocument[]} */
        this.selectedDocuments = [];
    }


    connectedCallback() {
        this.bindEvents();
        this.endPoint = this.dataset.streamUrl;
        this.logoutUrl = this.dataset.logoutUrl;
        if (this.messages.length) this.dataset.state = STATE.CONVERSATION;
    }


    bindEvents() {
        this.formElement?.addEventListener("submit", this.handleSubmit);

        listenEvent(Events.SELECTED_DOCS_CHANGE, (evt) => {
            this.selectedDocuments = evt.detail;
        });

        listenEvent(Events.STOP_STREAMING, () => {
            this.dataset.status = STATUS.STOPPED;
            this.currentStream?.socket.close();
        });
    }


    get formElement() {
        if (!this._formElement || !document.body.contains(this._formElement)) {
            this._formElement = /** @type {HTMLFormElement} */ (
                document.querySelector("#chats-form")
            );
        }
        return this._formElement;
    }


    get messageContainer() {
        if (!this._messageContainer || !this.contains(this._messageContainer)) {
            this._messageContainer = /** @type {HTMLElement} */ (
                this.querySelector("#chat-message-list")
            );
        }
        return this._messageContainer;
    }


    get messageInput() {
        if (!this._messageInput || !document.body.contains(this._messageInput)) {
            this._messageInput = /** @type {import("./message-input").MessageInput} */ (
                document.querySelector("ids-message-input")
            );
        }
        return this._messageInput;
    }


    /**
     * Returns the selected llm
     * @returns {string} Selected LLM
     */
    get llm() {
        return /** @type {HTMLInputElement | null}*/ (
          document.querySelector("#llm-selector")
        )?.value || "";
    }


    /**
     * Returns the last chat message
     * @returns {ChatMessage | null} Last chat message
     */
    get lastMessage() {
        const messages = this.messages;

        if (!messages?.length) return null;

        return messages[messages.length - 1];
    }


    /**
     * Returns the message element specified by ID
     * @param {string} id ChatMessage ID
     * @returns {ChatMessage | null} Message element
     */
    getMessage(id) {
        // switch to data-id on chat-message objects
        return /** @type {ChatMessage} */ (
            this.querySelector(`#chat-message-${id}`)
        );
    }


    /**
     * A list of all message elements
     * @returns {NodeListOf<ChatMessage>} Message element
     */
    get messages() {
        return /** @type {NodeListOf<ChatMessage>} */ (
            this.querySelectorAll("chat-message")
        );
    }


    handleSubmit = async (/** @type {Event} */evt) => {
        evt.preventDefault();

        if (!this.messageInput || this.currentStream) return;

        const text = this.messageInput.getValue();
        const hasContent = Boolean(text || this.messageInput.hasUploadedFiles());

        if (!hasContent) return;

        // TBC
        let activities = Array();
        if (this.selectedDocuments.length) {
            this.selectedDocuments.forEach(document => {activities.push(document.name)})
        }

        this.messageInput.reset(true);
        this.messageInput.collapse();

        this.dataset.state = STATE.CONVERSATION;

        emitEvent(Events.START_STREAMING);

        this.startStream({
            text: text,
            documents: this.selectedDocuments,
            llm: this.llm,
            toolId: getActiveToolId(),
            activities: activities,
        });
    };


    /**
     * Streams an LLM response
     * @param {StreamOptions} options
     */
    startStream = (options) => {
        const {
            text,
            documents,
            llm,
            toolId,
            activities,
        } = options;

        if (!this.endPoint) return console.error("Missing Endpoint");

        this.currentStream = {
            socket: new WebSocket(this.endPoint),
            messageId: "",
        }

        const webSocket = this.currentStream.socket;

        webSocket.onopen = () => {
            webSocket.send(
                JSON.stringify({
                    message: text,
                    sessionId: this.dataset.sessionId,
                    selectedFiles: documents,
                    activities: activities,
                    llm,
                    selectedTool: toolId,
                }),
            );
            this.dataset.status = STATUS.STREAMING;
            emitEvent(Events.CHAT_RESPONSE_START);
            emitEvent(Events.SCROLL_TO_BOTTOM, {source:this, force:true});
        };

        webSocket.onmessage = (evt) => {
            let response;
            try {
                response = JSON.parse(evt.data);
            } catch (err) {
                console.error("Error getting JSON response", err);
                return;
            }

            const data = response.data;

            switch (response.type) {
                case "message_created":
                    this.handleMessageCreated(data);
                    break;

                case "message_update":
                    this.handleMessageUpdate(data);
                    break;

                case "message_complete":
                    this.handleMessageComplete(data);
                    break;

                case "session-id":
                    this.dataset.sessionId = data;
                    break;

                case "message_activity":
                    this.handleMessageActivity(data);
                    break;

                case "auth_expired":
                    this.handleAuthExpired();
                    break;

                case "error":
                    this.handleError(data);
                    break;
            }
        };

        webSocket.onclose = () => {
            if (this.dataset.status !== STATUS.STOPPED) {
                this.dataset.status = STATUS.COMPLETE;
            }

            if (!this.currentStream) return;

            this.getMessage(this.currentStream.messageId)?.hideLoading();
            this.currentStream = null;
        };

        webSocket.onerror = () => {
            this.dataset.status = STATUS.ERROR;
            const error_message = "There was a problem. Please try sending this message again."

            if (!this.currentStream) return console.error(error_message);

            const message = this.getMessage(this.currentStream.messageId);
            message?.showError(error_message);
            this.currentStream = null;
        };
    };


    /**
     * Handle User / LLM message creation
     * @param {MessageCreatedResponse} response
     */
    handleMessageCreated(response) {
        if (!this.messageContainer) return console.error("Missing message container");
        if (!this.currentStream) return console.error("No active stream");

        const html = sanitizeHtml(response.html);
        this.messageContainer.insertAdjacentHTML("beforeend", html);

        if (response.chat_message_role === "ai") {
            this.currentStream.messageId = response.chat_message_id;
            this.getMessage(response.chat_message_id)?.focus();
        }
    }


    /**
     * Handle LLM message chunks
     * @param {MessageUpdateResponse} response
     */
    handleMessageUpdate(response) {
        const message = this.getMessage(response.chat_message_id);

        if (!message) return;

        message.updateContent(response.html, response.sr_text);
    }


    /**
     * Handle LLM message completion
     * @param {MessageCompleteResponse} response
     */
    handleMessageComplete(response) {
        const message = this.getMessage(response.chat_message_id);

        if (!message) return;

        message.complete(response.html);

        emitEvent(Events.CHAT_RESPONSE_END, {
            title: response.title,
            session_id: response.session_id,
            is_new_chat: this.messages.length == 2,
        })
    }


    /**
     * Handle LLM activity events
     * @param {MessageActivityResponse} response
     */
    handleMessageActivity(response) {
        const message = this.getMessage(response.chat_message_id);
        message?.showLoading(response.activity_event_message);
    }


    /**
     * Handle consumer errors
     * @param {string} response Error message
     */
    handleError(response) {
        if (this.currentStream) {
            const message = this.getMessage(this.currentStream.messageId);
            message?.showError(response);
        }
        console.error(`Stream error: ${response}`);
        emitEvent(Events.CHAT_RESPONSE_ERROR);
    }


    /**
     * Handle auth expiry events
     */
    handleAuthExpired() {
        if (this.logoutUrl) {
            window.location.href = this.logoutUrl;
        } else {
            this.lastMessage?.showError("Your session has expired.");
        }
    }
}
customElements.define("chat-controller", ChatController);

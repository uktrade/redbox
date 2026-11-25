// @ts-check

import { emitEvent, Events, listenEvent } from "../../../interaction_design_system/ids/events";
import { getActiveToolId, sanitizeHtml } from "../../utils";
import { saveDraft, loadDraft, deleteDraft } from "../../utils/draft-prompt-storage";
import { ChatMessage } from "./chat-message";
import htmx from "htmx.org";


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
        this.reauthenticateUrl = this.dataset.reauthenticateUrl;
        if (this.messages.length) this.dataset.state = STATE.CONVERSATION;
        void this.restoreDraft();
    }


    disconnectedCallback () {
        this.formElement?.removeEventListener(
            "submit",
            this.handleSubmit
        );
    }


    async restoreDraft () {
        const context = this.getDraftContext();
        const draftText = await loadDraft(context);
        if (!draftText || !this.isConnected) return

        const hasQuery = new URLSearchParams(window.location.search).has("q");
        const restored = this.messageInput?.setValue(draftText, hasQuery);
        if (restored) {
            await deleteDraft(context);
        }
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


    getDraftContext () {
        const userId = this.dataset.userId || "";
        const toolSlug = this.dataset.toolSlug || "general";
        const chatId = this.dataset.sessionId || "new";

        return {
            userId,
            chatKey: `${toolSlug}:${chatId}`
        }
    }


    handleSubmit = async (/** @type {Event} */evt) => {
        evt.preventDefault();

        if (!this.messageInput || this.currentStream) return;

        const messageText = this.messageInput.getValue();
        const hasContent = Boolean(messageText || this.messageInput.hasUploadedFiles());

        if (!hasContent) return;

        let activities = Array();
        if (this.selectedDocuments.length) {
            this.selectedDocuments.forEach(document => {activities.push(document.name)})
        }

        this.dataset.state = STATE.CONVERSATION;

        emitEvent(Events.START_STREAMING);

        this.startStream({
            text: messageText,
            documents: this.selectedDocuments.map(doc => doc.id),
            llm: this.llm,
            toolId: getActiveToolId(),
            activities: activities,
            draftContext: this.getDraftContext(),
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
            draftContext,
        } = options;

        if (!this.endPoint) return console.error("Missing Endpoint");

        this.currentStream = {
            socket: new WebSocket(this.endPoint),
            messageId: "",
            title: "",
            messageAccepted: false,
            draftContext,
        }

        const webSocket = this.currentStream.socket;

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
                case "chat_ready":
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
                    emitEvent(Events.SCROLL_TO_BOTTOM, { source: this, force: true });
                    break;

                case "message_created":
                    this.handleMessageCreated(data);

                    if (
                        data.chat_message_role === "user" &&
                        this.currentStream &&
                        !this.currentStream?.messageAccepted
                    ) {

                        const draftContext = this.currentStream.draftContext;
                        this.currentStream.messageAccepted = true;
                        this.messageInput.reset(true);
                        this.messageInput.collapse();
                        void deleteDraft(draftContext);
                    }

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

        webSocket.onclose = (event) => {
            if (this.authRedirecting) return;

            if (event.code === 4001) {
                this.handleAuthExpired();
                return;
            }

            if (!this.currentStream) return;

            if (event.code !== 1000 && this.dataset.status !== STATUS.STOPPED){
                this.dataset.status = STATUS.ERROR;
                this.handleError(
                    "The connnection was lost. Please try again"
                );

                this.currentStream = null;
                return;
            }

            if (this.dataset.status === STATUS.ERROR) {
                this.currentStream = null;
                return;
            }

            this.getMessage(this.currentStream.messageId)?.setComplete();
            emitEvent(Events.CHAT_RESPONSE_END, {
                title: this.currentStream.title,
                session_id: this.dataset.sessionId || "",
                is_new_chat: this.messages.length == 2,
            })

            this.currentStream = null;
        };

        webSocket.onerror = () => {
            console.error("Websocket connection error")
        }
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

        const message = this.getMessage(response.chat_message_id);

        if (message) htmx.process(message);

        if (response.chat_message_role === "ai") {
            this.currentStream.messageId = response.chat_message_id;
            message?.focus();
            message?.setStreaming()
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
        message?.complete(response.html);
        if (this.currentStream) this.currentStream.title = response.title;
        emitEvent(Events.SCROLL_TO_BOTTOM, {source:this, force:true});
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
    handleError (response) {
        this.dataset.status = STATUS.ERROR;
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
    async handleAuthExpired () {
        if (this.authRedirecting) return;

        if (!this.reauthenticateUrl) {
            this.lastMessage?.showError(
                "Your session has expired. Please sign in again."
            );
            return;
        }

        this.authRedirecting = true;

        const url = new URL(
            this.reauthenticateUrl,
            window.location.origin
        );
        const chatId = this.dataset.sessionId;
        let nextPath = window.location.pathname;

        if (chatId) {
            const prefix = this.dataset.toolSlug
                ? `/tools/${this.dataset.toolSlug}/chats/`
                : "/chats/";

            nextPath = `${prefix}${chatId}/`;
        }

        url.searchParams.set(
            "next",
            nextPath + window.location.search
        )

        try {
            const messageText = this.messageInput?.getValue(false);
            if (messageText?.trim()) {
                const saved = await saveDraft({
                    ...this.getDraftContext(),
                    text: messageText
                });
            }
        } finally {
            window.location.assign(url.toString());
        }

    }
}
customElements.define("chat-controller", ChatController);

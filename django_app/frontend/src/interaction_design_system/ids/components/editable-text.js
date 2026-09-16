// @ts-check

import { getAttributeOrDefault, getCsrfToken, hideElement, showElement } from "../../../js/utils";
import { emitEvent, Events, listenEvent } from "../events";

class EditableText extends HTMLElement {
    _csrfToken = "";

    constructor() {
        super();

        this.postUrl = this.getAttribute("post-url");
        this.deleteUrl = this.getAttribute("delete-url");
        this.objectId = this.getAttribute("object-id") || undefined;

        let senderId = this.getAttribute("sender-id");
        if (!senderId) {
            senderId = crypto.randomUUID();
            this.setAttribute("sender-id", senderId);
        }
        this.senderId = senderId;
    }

    connectedCallback() {
        this.displayEl = /** @type {HTMLElement} */ (
            this.querySelector(getAttributeOrDefault(this, "display-selector", ".display"))
        );
        this.textEl = /** @type {HTMLElement} */ (
            this.querySelector(getAttributeOrDefault(this, "text-selector", ".title-text"))
        );
        this.inputWrapper = /** @type {HTMLElement} */ (
            this.querySelector(getAttributeOrDefault(this, "input-wrapper-selector", ".edit-input-wrapper"))
        );
        this.inputEl = /** @type {HTMLInputElement} */ (
            this.querySelector(getAttributeOrDefault(this, "input-selector", ".edit-input"))
        );
        this.editTrigger = /** @type {HTMLElement} */ (
            this.querySelector(getAttributeOrDefault(this, "edit-selector", ".edit-trigger"))
        );
        this.deleteTrigger = /** @type {HTMLElement} */ (
            this.querySelector(getAttributeOrDefault(this, "delete-selector", ".delete-trigger"))
        );

        this.#bindEvents();
        this.#listenForExternalUpdates();
    }


    /**
     * Function to bind edit and delete events and send post requests.
    */
    #bindEvents() {
        if (this.editTrigger) {
            this.editTrigger.addEventListener("click", () => this.#enterEditMode());
        }

        if (this.inputEl) {
            this.inputEl.addEventListener("keydown", (evt) => {
                if (!this.inputEl) return false;

                switch (/** @type {KeyboardEvent} */ (evt).key) {
                    case "Escape":
                        this.inputEl.value = this.inputEl.dataset.title || "";
                        this.#exitEditMode();
                        return true;
                    case "Enter":
                        evt.preventDefault();
                        this.#save();
                        return true;
                    default:
                        return true;
                }
            });

            this.inputEl.addEventListener("change", () => this.#save());
            this.inputEl.addEventListener("blur", () => this.#save());
        }

        if (this.deleteTrigger && this.deleteUrl) {
            this.deleteTrigger.addEventListener("click", () => this.#delete());
        }
    }


    /**
     * Show input element and hide display element
    */
    #enterEditMode() {
        if (!this.inputEl || !this.textEl || !this.displayEl || !this.inputWrapper) return;
        this.inputEl.dataset.title = this.textEl.innerText || "";
        this.inputEl.value = this.textEl.title || this.textEl.innerText || "";
        hideElement(this.displayEl);
        showElement(this.inputWrapper);
        this.inputEl.focus();
    }


    /**
     * Show display element and hide input element
    */
    #exitEditMode() {
        if (!this.inputEl || !this.textEl || !this.displayEl || !this.inputWrapper) return;
        this.textEl.innerText = this.inputEl.value;
        this.textEl.title = this.inputEl.value;
        hideElement(this.inputWrapper);
        showElement(this.displayEl);
        this.editTrigger?.focus();
    }


    /**
     * Initiate post request to save changes
    */
    async #save() {
        if (!this.inputEl || !this.textEl) return;
        const newValue = this.inputEl.value;
        if (!newValue) return;
        if (newValue === this.textEl.innerText) {
            this.#exitEditMode();
            return;
        }

        if (!this.postUrl) return;

        try {
            await fetch(this.postUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json", "X-CSRFToken": this.csrfToken },
                body:JSON.stringify({value: newValue}),
            });
        } catch (err) {
            console.error("Save failed", err);
            this.inputEl.value = this.inputEl.dataset.title || "";
        } finally {
            this.#exitEditMode();

            emitEvent(Events.EDITABLE_TEXT_CHANGE, {
                sender_id: this.senderId,
                object_id: this.objectId,
                value: newValue,
            })
        }
    }


    async #delete() {
        if (!this.deleteUrl) return;

        try {
            await fetch(this.deleteUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json", "X-CSRFToken": this.csrfToken },
            });
        } catch (err) {
            console.error("Delete failed", err);
        } finally {
            emitEvent(Events.EDITABLE_TEXT_DELETE, {
                sender_id: this.senderId,
                object_id: this.objectId,
            });
            this.remove();
        }
    }


    /**
     * Listen for external events of the same type and update value accordingly
    */
    #listenForExternalUpdates() {
        listenEvent(Events.EDITABLE_TEXT_CHANGE, (evt) => {
            const {sender_id: senderId, object_id: objectId, value: newValue} = evt.detail;

            if (senderId === this.senderId) return;
            if (!objectId || objectId !== this.objectId) return;
            if (typeof newValue === "string" && this.textEl) this.textEl.innerText = newValue;
        });
    }


    get csrfToken() {
        if (!this._csrfToken) this._csrfToken = getCsrfToken();
        return this._csrfToken;
    }
}
customElements.define("ids-editable-text", EditableText);

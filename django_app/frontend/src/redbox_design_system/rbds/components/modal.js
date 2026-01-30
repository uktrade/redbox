// @ts-check

import { disableNoScroll, enableNoScroll } from "./no-scroll";

export class Modal extends HTMLElement {

    connectedCallback() {
        this.#initialiseModal();
    }


    get dialog() {
        return this.querySelector("dialog");
    }


    /**
     * Returns the dialog ID
     * @returns {String | null | undefined} dialog ID
     */
    get dialogId () {
        return this._dialogId;
    }


    /**
     * Returns the dialog ID
     * @param {String | null} id dialog ID
     */
    set dialogId (id) {
        this._dialogId = id;
    }


    #initialiseModal() {
        if (!this.dialog) return;

        const openElement = this.querySelector(".modal-open");
        const closeElements = this.querySelectorAll(".modal-close");

        openElement?.addEventListener("click", (evt) => {
            evt.preventDefault();
            if (!this.dialog) return;

            this.dialog.showModal();
            this.dialogId = enableNoScroll();
        });

        closeElements.forEach((element) => {
            element.addEventListener("click", (evt) => {
                evt.preventDefault();
                if (!this.dialog) return;

                this.dialog.close();
                disableNoScroll(this.dialogId);
            });
        });

        this.dialog.addEventListener("click", (event) => {
            const target = /** @type {HTMLDialogElement} */ (event.target);
            const rect = target.getBoundingClientRect();
            const isInDialog =
                rect.top <= event.clientY &&
                event.clientY <= rect.top + rect.height &&
                rect.left <= event.clientX &&
                event.clientX <= rect.left + rect.width;
            if (!isInDialog) {
                target.close();
                disableNoScroll(this.dialogId);
            }
        });
    }
}
customElements.define("rbds-modal", Modal);

// @ts-check

import { LoadingMessage } from "../../../interaction_design_system/ids/components";
import { hideElement, sanitizeHtml, showElement } from "../../utils";
import { StreamedContent } from "../streamed-content";

export class ChatMessage extends HTMLElement {
    constructor() {
        super();

        this.errorContainerSelector = ".govuk-error-summary";
        this.errorContentSelector = ".govuk-error-summary__title";
    }


    /**
     * Returns the container for the streamed response
     * @returns {StreamedContent} StreamedContent response container
     */
    get streamedContent() {
        return /** @type {StreamedContent} */ (
            this.querySelector("streamed-content")
        );
    }


    /**
     * Updates the content with the streamed response
     * @param {string} html Visual HTML
     * @param {string} srText Screen reader text
     */
    updateContent(html, srText = "") {
        this.streamedContent?.update(html, srText);
    }


    /**
     * Updates the streamed final HTML content
     * @param {string} html Final HTML
     */
    complete(html) {
        this.streamedContent?.complete(html);
        this.hideLoading();
    }


    /**
     * TBC - Show error element
     * @param {string} message Error message
     */
    showError(message) {
        const error = this.querySelector(this.errorContainerSelector);

        if (!error) return;

        showElement(error);

        const body = error.querySelector(this.errorContentSelector);
        if (body) body.innerHTML = sanitizeHtml(message);
    }


    /**
     * TBC - Hide error element
     */
    hideError() {
        const error = this.querySelector(this.errorContainerSelector);
        if (error) hideElement(error);
    }


    /**
    * Returns the activity element used for response feedback
    * @returns {LoadingMessage} Loading Message Activity element
    */
    get loadingElement() {
        // TODO: Check announcements, what happens during streaming?
        if (!this._loadingElement || !this.contains(this._loadingElement)) {
            this._loadingElement = /** @type {LoadingMessage} */ (
                this.querySelector('ids-loading-message')
            );
        }

        return this._loadingElement;
    }


    /**
    * Show message loading activity
    * @param {string} text loading activity text
    */
    showLoading(text="Loading") {
        this.loadingElement.setText(text);
        showElement(this.loadingElement);
    }


    /**
    * Hide message loading activity
    */
    hideLoading() {
        hideElement(this.loadingElement);
    }
}
customElements.define("chat-message", ChatMessage);

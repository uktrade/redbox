// @ts-check

import { focusFirstFocusable } from "../../../js/utils/dom-utils";

class CollapsibleMenu extends HTMLElement {
    constructor() {
        super();
        this.onDocumentClick = this.onDocumentClick.bind(this);
        this.onKeyPress = this.onKeyPress.bind(this);
        this.onToggle = this.onToggle.bind(this);
        this.menuItemSelector = ".ids-collapsible-menu-item";
    }

    connectedCallback() {
        this.addEventListener("focusout", this.onFocusOut)

        this.details = this.querySelector('details');
        if (!this.details) return;

        this.details.addEventListener('toggle', this.onToggle);
    }

    disconnectedCallback() {
        if (!this.details) return;
        this.details.removeEventListener('toggle', this.onToggle);
        document.removeEventListener('click', this.onDocumentClick);
        document.removeEventListener('keydown', this.onKeyPress);
    }


    get previousElement() {
        const previousElement = /** @type {HTMLElement} */ (
            document.activeElement?.previousElementSibling
        );
        if (previousElement && this.contains(previousElement)) {
            return previousElement;
        } else {
            return /** @type {HTMLElement} */ (this.firstElementChild);
        }
    }


    get nextElement() {
        const nextElement = /** @type {HTMLElement} */ (
            document.activeElement?.nextElementSibling
        );
        if (nextElement && this.contains(nextElement)) {
            return nextElement;
        } else {
            return null;
        }
    }


    /**
     * Attach/detach listeners on open/close
     */
    onToggle() {
        if (!this.details) return;

        if (this.details.open) {
            document.addEventListener('click', this.onDocumentClick);
            document.addEventListener('keydown', this.onKeyPress);
        } else {
            document.removeEventListener('click', this.onDocumentClick);
            document.removeEventListener('keydown', this.onKeyPress);
        }
    }


    /**
     * Handle click events
     * @param {Event} event event object
     */
    onDocumentClick(event) {
        if (!event || !this.details || !this.details.open) return;
        const target = /** @type {HTMLElement} */ (event.target);

        // Close if click outside this component
        if (!this.contains(target)) this.details.open = false;
    }


    /**
     * Handle focus out events
     * @param {FocusEvent} event event object
     */
    onFocusOut(event){
        if (!event || !this.details || !this.details.open) return;
        if (!this.contains(event.relatedTarget)){
            this.details.open = false;
        }
    }


    /**
     * Handle key press events
     * @param {KeyboardEvent} event event object
     */
    onKeyPress(event) {
        if (!event || !this.details) return;
        switch (event.key) {
            case "Escape":
            case "Esc":
                this.details.open = false;
                const firstElement = /** @type {HTMLElement} */ (this.firstElementChild);
                focusFirstFocusable(firstElement);
                break;
            case "ArrowUp":
                if (this.details.open) focusFirstFocusable(this.previousElement);
                break;
            case "ArrowDown":
                if (this.details.open) focusFirstFocusable(this.nextElement);
                break;
            default:
                return;
        }
    }
}
customElements.define('ids-collapsible-menu', CollapsibleMenu);

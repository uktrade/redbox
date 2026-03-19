// @ts-check

class IdsCollapsibleMenu extends HTMLElement {
    constructor() {
        super();
        this.onDocumentClick = this.onDocumentClick.bind(this);
        this.onEscapeKey = this.onEscapeKey.bind(this);
        this.onToggle = this.onToggle.bind(this);
    }
    connectedCallback() {
        this.details = this.querySelector('details');
        if (!this.details) return;

        this.details.addEventListener('toggle', this.onToggle);
    }

    disconnectedCallback() {
        if (!this.details) return;
        this.details.removeEventListener('toggle', this.onToggle);
        document.removeEventListener('click', this.onDocumentClick);
        document.removeEventListener('keydown', this.onEscapeKey);
    }

    /**
     * Attach/detach listeners on open/close
     */
    onToggle() {
        if (!this.details) return;

        if (this.details.open) {
            document.addEventListener('click', this.onDocumentClick);
            document.addEventListener('keydown', this.onEscapeKey);
        } else {
            document.removeEventListener('click', this.onDocumentClick);
            document.removeEventListener('keydown', this.onEscapeKey);
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
     * Handle escape key events
     * @param {KeyboardEvent} event event object
     */
    onEscapeKey(event) {
        if (!event || !this.details) return;
        if (event.key === 'Escape' || event.key === 'Esc') {
            this.details.open = false;

            // Move focus back to the summary element
            const summary = this.details.querySelector('summary');
            summary?.focus();
        }
    }
}
customElements.define('ids-collapsible-menu', IdsCollapsibleMenu);

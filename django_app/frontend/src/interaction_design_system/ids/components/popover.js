// @ts-check

import { hideElement, isHidden, showElement, visuallyHideElement } from "../../../js/utils";

export class Popover extends HTMLElement {
    constructor() {
        super();

        this.isPinned = false;

        this.onKeyDown = this.onKeyDown.bind(this);
        this.onDocumentClick = this.onDocumentClick.bind(this);
        this.onFocus = this.onFocus.bind(this);
    }


    connectedCallback() {
        this.initialise();
    }


    disconnectedCallback() {
        this.removePanelListeners();
    }


    initialise() {
        /** @type {HTMLElement | null} */
        this.trigger = this.querySelector(".ids-popover__trigger");

        /** @type {HTMLElement | null} */
        this.panel = this.querySelector(".ids-popover__panel");

        /** @type {HTMLElement | null} */
        this.closeButton = this.querySelector(".ids-popover__close");

        this.openOnHover = this.dataset.openOnHover?.toLowerCase() !== "false";

        this.#bindEvents();
    }


    #bindEvents() {
        if (!this.trigger || !this.panel) return;

        // Click to toggle open/close and pinned state
        this.trigger.addEventListener("click", (evt) => {
            evt.preventDefault();

            if (!this.isOpen()) return this.open(true);

            this.isPinned ? this.close() : this.pin();
        });

        // Close button (optional)
        this.closeButton?.addEventListener("click", (evt) => {
            evt.preventDefault();
            this.close();
        });

        // Hover/Focus behaviour
        this.trigger.addEventListener("pointerenter", () => {
            if (!this.isOpen() && this.openOnHover) this.open();
        });
        this.trigger.addEventListener("pointerleave", () => {
            if (!this.isPinned && this.openOnHover) this.scheduleClose();
        });

        this.trigger.addEventListener("focus", (evt) => {
            if (!this.isOpen() && this.openOnHover) this.open(this.openOnHover);
        });

        this.panel.addEventListener("pointerleave", () => {
            if (!this.isPinned) this.scheduleClose();
        });
    }


    isOpen() {
        return !isHidden(this.panel);
    }


    addPanelListeners() {
        document.addEventListener("click", this.onDocumentClick);
        document.addEventListener("keydown", this.onKeyDown);
        this.addEventListener("focusin", this.onFocus);
    }


    removePanelListeners() {
        document.removeEventListener("click", this.onDocumentClick);
        document.removeEventListener("keydown", this.onKeyDown);
        this.removeEventListener("focusin", this.onFocus);
    }


    /**
     * Open the popover panel
     * @param {number} viewportGap Minimum px distance from viewport edge
     * @param {number} triggerGap Distance between the popover and trigger
    */
    positionPanel(viewportGap=5, triggerGap=5) {
        if (!this.trigger || !this.panel) return;

        const trigger = this.trigger.getBoundingClientRect();
        const panel = this.panel.getBoundingClientRect();
        const container = this.getBoundingClientRect();

        let viewportLeft;

        // Default: align panel left edge with trigger
        viewportLeft = trigger.left;

        // Overflow right
        if (viewportLeft + panel.width > window.innerWidth - viewportGap) {
            viewportLeft = trigger.right - panel.width;
        }

        // Still overflowing left
        if (viewportLeft < viewportGap) viewportLeft = viewportGap;

        let viewportTop;

        // Default: below trigger
        viewportTop = trigger.bottom + triggerGap;

        // Flip above if needed
        if (viewportTop + panel.height > window.innerHeight - viewportGap) {
            viewportTop = trigger.top - panel.height - triggerGap;
        }

        // Clamp
        if (viewportTop < viewportGap) viewportTop = viewportGap;


        // Convert viewport coords -> relative to ids-popover
        this.panel.style.left = `${viewportLeft - container.left}px`;
        this.panel.style.top = `${viewportTop - container.top}px`;
    }


    /**
     * Open the popover panel
     * @param {boolean} pin
     */
    open(pin=false) {
        if (!this.trigger || !this.panel) return;

        showElement(this.panel);

        this.positionPanel();

        this.trigger.setAttribute("aria-expanded", "true");
        this.panel.dataset.open = "true";

        if (pin) this.pin();

        this.addPanelListeners();
    }


    close() {
        if (!this.trigger || !this.panel) return;

        hideElement(this.panel);
        this.trigger.setAttribute("aria-expanded", "false");
        this.panel.dataset.open = "false";
        this.removePanelListeners();
        this.unpin();
    }


    pin() {
        this.isPinned = true;
    }


    unpin() {
        this.isPinned = false;
    }


    /**
     * Schedule popover closure
     *
     * Small delay prevents flicker when moving between trigger/panel
     * @param {number} delay delay in milliseconds
     */
    scheduleClose(delay=100) {
        setTimeout(() => {
            if (this.isPinned) return;
            if (!this.trigger || !this.panel) return;

            const hovered = this.trigger.matches(":hover") || this.panel.matches(":hover");
            const focused = this.contains(document.activeElement);

            if (!hovered && !focused) {
                this.close();
            }
        }, delay);
    }


    /**
     * Handle escape key events
     * @param {KeyboardEvent} evt event object
     */
    onKeyDown(evt) {
        if (evt.key !== "Escape" && evt.key !== "Esc") return;
        if (!this.panel || !this.isOpen()) return;

        this.close();
    }


    /**
     * Handle click events
     * @param {Event} evt event object
     */
    onDocumentClick(evt) {
        if (!this.panel || !this.trigger) return;

        const target = /** @type {Node} */ (evt.target);

        // click inside component > ignore
        if (this.contains(target)) return;

        // click outside > close
        this.close();
    }


    /**
     * Handle focus events
     * @param {Event} evt event object
     */
    onFocus(evt) {
        if (!this.panel || !this.trigger) return;

        const target = /** @type {Node} */ (evt.target);

        // close panel when open on loss of focus
        if (this.isOpen() && !this.contains(target)) this.close();
    }
}
customElements.define("ids-popover", Popover);

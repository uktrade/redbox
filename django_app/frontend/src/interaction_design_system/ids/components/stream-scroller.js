// @ts-check

import { Events, listenEvent } from "../events";

export class StreamScroller extends HTMLElement {
    constructor() {
        super();
        this.autoScrollEnabled = true;
        this.programmaticScroll = false;
        this.scrollPending = false;
        this._anchor = null;
        this._observer = null;
    }

    connectedCallback() {
        this.scrollContainer = this.getScrollParent();
        this.#bindScrollEvents();
        this.#setupAnchor();
        this.#setupIntersectionObserver();
    }

    disconnectedCallback() {
        this.scrollContainer?.removeEventListener("scroll", () => this.#onScroll());
        this._observer?.disconnect();
    }


    #bindScrollEvents() {
        this.scrollContainer?.addEventListener("scroll", () => this.#onScroll(), { passive: true });
        listenEvent(Events.SCROLL_TO_BOTTOM, (evt) => this.scheduleScroll(evt.detail));
    }


    #onScroll() {
        if (this.programmaticScroll) return;
        console.log("isatbottom", this.#isAtBottom());
        this.autoScrollEnabled = this.#isAtBottom();
    }

    #setupAnchor() {
        // Auto-create anchor if missing
        if (!this._anchor) {
            this._anchor = document.createElement("div");
            this._anchor.className = "ids-scroll-anchor";
            this.appendChild(this._anchor);
        }
    }


    #setupIntersectionObserver() {
        if (!this._anchor) return;

        this._observer = new IntersectionObserver(([entry]) => {
            // Anchor moved out of view due to content → scroll if allowed
            if (!entry.isIntersecting && !this.programmaticScroll && this.autoScrollEnabled) {
                this.scheduleScroll();
            }

            // Update autoScrollEnabled based on visibility
            this.autoScrollEnabled = entry.isIntersecting;

        }, { root: null, threshold: 0 });

        this._observer.observe(this._anchor);
    }


    #isAtBottom(threshold = 10) {
        if (!this.scrollContainer) return false;

        let scrollElement;
        if (this.scrollContainer instanceof Document) {
            scrollElement = this.scrollContainer.documentElement;
        } else {
            scrollElement = this.scrollContainer;
        }

        return scrollElement.scrollHeight - scrollElement.scrollTop - scrollElement.clientHeight <= threshold;
    }


    /**
     * Finds the nearest scrollable ancestor
     * @param {HTMLElement} element
     * @returns {HTMLElement | Element | Document | undefined}
     */
    getScrollParent(element = this) {
        let parent = element.parentElement;

        while (parent) {
            const style = getComputedStyle(parent);

            const overflowY = style.overflowY;
            const canScroll =
                overflowY === "auto" ||
                overflowY === "scroll" ||
                overflowY === "overlay";

            if (canScroll && parent.scrollHeight > parent.clientHeight) {
                if (parent == document.documentElement) return document;
                return parent;
            }

            parent = parent.parentElement;
        }

        return document.scrollingElement || document.documentElement;
    }


    #scrollToBottom() {
        if (!this.scrollContainer) return;

        this.programmaticScroll = true;

        // Move sentinel to bottom before scrolling
        if (this._anchor) this.appendChild(this._anchor);

        let scrollElement;
        if (this.scrollContainer instanceof Document) {
            scrollElement = this.scrollContainer.documentElement;
        } else {
            scrollElement = this.scrollContainer;
        }
        scrollElement.scrollTop = scrollElement.scrollHeight;

        this.programmaticScroll = false;
    }


    scheduleScroll({ force = false } = {}) {
        if (!this.autoScrollEnabled && !force) return;
        if (this.scrollPending) return;

        if (force) this.programmaticScroll = true;

        this.scrollPending = true;
        this.#scrollToBottom();
        this.scrollPending = false;
    }
}

customElements.define("ids-stream-scroller", StreamScroller);

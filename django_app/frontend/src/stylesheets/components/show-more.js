// @ts-check

import { hideElement, showElement, getNumericAttr } from "../../js/utils";

class ShowMore extends HTMLElement {
    visibleCount = 5;
    labelText = "Show more...";


    connectedCallback() {
        this.visibleCount = getNumericAttr(this, 'visible-count', this.visibleCount);
        this.labelText = this.getAttribute('label-text') || this.labelText;

        if (!this.container) return;

        this.#updateVisibleItems();
        this.observer = new MutationObserver((mutations, observer) => {
            for (const mutation of mutations) {
                if (mutation.target.nodeName !== this.nodeName) this.#updateVisibleItems();
            }
        });

        this.observer.observe(this.container, {
            childList: true,
            subtree: true,
        });
    }

    disconnectedCallback() {
        if (this.observer) this.observer.disconnect();
    }


    /**
     * Getter for up-to-date container element
    */
    get container() {
        const containerSelector = this.getAttribute('container-selector');
        if (!containerSelector) {
            console.warn('ShowMore: container-selector attribute is missing');
            return null;
        }

        const container = document.querySelector(containerSelector);
        if (!container) {
            console.warn(`ShowMore: container not found for selector '${containerSelector}'`);
            return null;
        }
        return container;
    }


    /**
     * Getter for up-to-date container items
    */
    get items() {
        const itemSelector = this.getAttribute('item-selector');
        if (!itemSelector) {
            console.warn('ShowMore: item-selector attribute is missing');
            return null;
        }
        const items = /** @type {NodeListOf<HTMLElement>} */ (
            this.container?.querySelectorAll(itemSelector)
        );
        return items;
    }


    /**
     * Function to update the list of visible items when elements are added/removed.
     * @param {NodeListOf<HTMLElement> | null} items - list of item elements
    */
    #updateVisibleItems(items = this.items) {
        if (!items) return;
        if (this.isExpanded && items.length > this.visibleCount) return;

        if (items.length <= this.visibleCount) {
            this.isExpanded = null;
            items.forEach((item) => showElement(item));
        } else {
            items.forEach((item, index) => {
                this.isExpanded = false;
                if (index >= this.visibleCount) {
                    hideElement(item);
                } else {
                    showElement(item);
                }
            });
        }
        this.#showHideClickElement(items, this.visibleCount, this.labelText);
    }


    /**
     * @param {NodeListOf<HTMLElement>} items - list of item elements
    */
    #showHideClickElement(items, visibleCount, labelText) {
        let ClickElement = this.querySelector("a");

        if (!ClickElement && items.length > visibleCount) {
            // TODO: Add support for custom click element?
            ClickElement = document.createElement("a");
            ClickElement.textContent = labelText;
            ClickElement.classList.add(
                "show-more-link",
                "govuk-link",
                "govuk-link--no-visited-state",
                "govuk-link--no-underline"
            );

            this.appendChild(ClickElement);

            ClickElement.addEventListener("click", (evt) => {
                this.isExpanded = true;
                items.forEach((item) => showElement(item));

                if (evt.currentTarget) {
                    const target = /** @type {HTMLElement} */ (
                        evt.currentTarget
                    );
                    target.remove();
                }
            });
        }

        if (ClickElement && items.length <= visibleCount) ClickElement.remove();
    }

    /**
     * Getter for isExpanded property for aria-expanded
    */
    get isExpanded() {
        const value = this.getAttribute("aria-expanded");
        if (value == null) return null;
        return value === "true";
    }

    /**
     * Setter for isExpanded property for aria-expanded
     * @param {boolean | null | undefined} value - Name of attribute
    */
    set isExpanded(value) {
        if (value === null || value === undefined) {
            this.removeAttribute("aria-expanded");
        } else {
            this.setAttribute("aria-expanded", value ? "true" : "false");
        }
    }
}
customElements.define("show-more", ShowMore);

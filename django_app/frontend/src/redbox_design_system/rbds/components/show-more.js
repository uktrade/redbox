// @ts-check

import htmx from "htmx.org";
import { hideElement, showElement, getNumericAttr } from "../../../js/utils";

export class ShowMore extends HTMLElement {
    _visibleCount = 5;
    _showMoreLabelText = "Show more...";
    _showLessLabelText = "";

    connectedCallback() {
        if (!this.container) return;
        htmx.onLoad(() => this.#updateVisibleItems());
        this.initObserver();
    }

    disconnectedCallback() {
        if (this.observer) this.observer.disconnect();
    }


    /**
     * Setup observer for tracking changes to container and elements
    */
    initObserver() {
        if (!this.container) return;

        if (this.observer) this.observer.disconnect();

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


    /**
     * Getter for up-to-date visible elements limit
    */
    get visibleCount() {
        return getNumericAttr(this, 'visible-count', this._visibleCount);
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
     * Getter for up-to-date label text
    */
    get showMoreLabelText() {
        return this.getAttribute('show-more-label') || this._showMoreLabelText;
    }


    /**
     * Getter for enabling show less button functionality
    */
    get showLessLabelText() {
        return this.getAttribute('show-less-label') || this._showLessLabelText;
    }


    /**
     * Function to update the list of visible items when elements are added/removed.
     * @param {NodeListOf<HTMLElement> | null} items - list of item elements
    */
    #updateVisibleItems(items = this.items) {
        if (!items) return;
        if (this.isExpanded && items.length > this.visibleCount) {
            this.#addShowLess(items);
            return;
        }

        if (items.length <= this.visibleCount) {
            this.#showItems(items, null);
        } else {
            this.#hideItems(items);
        }
        this.#addShowMore(items);
    }


    /**
     * Function to create show more/less element with provided text
     * @param {NodeListOf<HTMLElement> | null} items - list of item elements
    */
    #addShowMoreOrLess(
        items=this.items,
        visibleCount=this.visibleCount,
        labelText=this.showMoreLabelText,
    ) {
        if (!items || !labelText) return;

        let ClickElement = this.querySelector("a");

        if (!ClickElement && items.length > visibleCount) {
            ClickElement = this.#getOrCreateClickElement(labelText);
            this.appendChild(ClickElement);

            ClickElement.addEventListener("click", (evt) => {
                if (labelText === this.showMoreLabelText) {
                    this.#showItems(this.items, true);
                } else {
                    this.#hideItems();
                }
                if (!evt.currentTarget) return;

                const target = /** @type {HTMLElement} */ (evt.currentTarget);
                hideElement(target);

                if (labelText === this.showMoreLabelText) {
                    this.#addShowLess();
                } else {
                    this.#addShowMore();
                }
            });
        }
        if (ClickElement && items.length <= visibleCount) ClickElement.remove();
    }


    /**
     * Function to add show more element and limit number of visible items to visibleCount
    */
    #addShowMore(items=this.items, visibleCount=this.visibleCount, showMorelabelText=this.showMoreLabelText) {
        this.#addShowMoreOrLess(items, visibleCount, showMorelabelText);
    }


    /**
     * Function to add show less element and show all elements in the container
    */
    #addShowLess(items=this.items, visibleCount=this.visibleCount, showLessLabelText=this.showLessLabelText) {
        this.#addShowMoreOrLess(items, visibleCount, showLessLabelText);
    }


    /**
     * Function to limit number of visible items to visibleCount
    */
    #hideItems(items=this.items) {
        this.isExpanded = false;
        items?.forEach((item, index) => {
            if (index >= this.visibleCount) {
                hideElement(item);
            } else {
                showElement(item);
            }
        });
    }


    /**
     * Function to show all items in the container
     * @param {boolean | null | undefined} expanded - aria-expanded value
    */
    #showItems(items=this.items, expanded=true) {
        this.isExpanded = expanded;
        items?.forEach((item) => showElement(item));
    }


    /**
     * Function to create click element for toggling show more/less functionality
     * Currently hardcoded to an anchor element
     * @param {string} labelText - label text for click element
    */
    #getOrCreateClickElement(labelText) {
        // TODO: Add support for custom click element?
        let clickElement = document.createElement("a");
        clickElement.textContent = labelText;
        clickElement.classList.add(
            "rbds-action-link",
            "govuk-link",
            "govuk-link--no-visited-state",
            "govuk-link--no-underline",
        );
        return clickElement;
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
customElements.define("rbds-show-more", ShowMore);

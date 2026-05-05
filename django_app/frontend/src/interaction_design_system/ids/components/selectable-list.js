// @ts-check

import { hideElement, isHidden, showElement } from "../../../js/utils";

export class SelectableList extends HTMLElement {
    constructor() {
        super();

        this._items = Array();
        this._searchQuery = "";
    }

    static get observedAttributes() {
        return ["data-highlight"];
    }

    get visibleItems() {
        return this._items.filter(item => !isHidden(item.row));
    }

    connectedCallback() {
        this.#cacheDom();
        this.#initItems();
        this.#bindEvents();
        this.#filter(); // initial render
    }

    attributeChangedCallback() {
        this.#filter();
    }

    // ---------- Public API ----------

    get highlightEnabled() {
        const val = this.dataset.highlight;
        return val === "" || val === "true";
    }

    get selectedIds() {
        return this._items
            .filter(item => item.checkbox?.checked)
            .map(item => item.checkbox.value);
    }

    // ---------- Setup ----------

    #cacheDom() {
        this.$search = this.querySelector("[data-ids-search]");
        this.$selectAll = this.querySelector("[data-ids-select-all]");
        this.$items = Array.from(this.querySelectorAll("[data-ids-item]"));
        this.selectedCounters = Array.from(this.querySelectorAll("[data-ids-selected-count]"));
    }


    #initItems() {
        if (!this.$items) return;

        this._items = this.$items.map(el => {
            const elem = /** @type {HTMLElement} */ (el);
            const row = /** @type {HTMLTableRowElement} */ (el.closest("tr"));
            const labelEls = Array.from(/** @type {NodeListOf<HTMLElement>} */ (
                    row.querySelectorAll("[data-ids-label]")
            ));

            labelEls.forEach(labelEl => {
                if (!labelEl.dataset.originalText) {
                    labelEl.dataset.originalText = labelEl.textContent || "";
                }
            });

            return {
                elem,
                row,
                checkbox: elem.querySelector('input[type="checkbox"]'),
                labelEls,
                aliases: (elem.dataset.aliases || "")
                    .split("|")
                    .map(a => a.trim())
                    .filter(Boolean),
            };
        });
    }


    #bindEvents() {
        this.$search?.addEventListener("input", (/** @type {Event} */ evt) => {
            const target = /** @type {HTMLInputElement} */ (evt.target);
            this._searchQuery = target.value;
            this.#filter();
        });

        this.$selectAll?.addEventListener("change", (/** @type {Event} */ evt) => {
            const target = /** @type {HTMLInputElement} */ (evt.target);

            const checked = target.checked;

            this.visibleItems.forEach(item => {
                if (item.checkbox) {
                    item.checkbox.checked = checked;
                }
            });

            this.#updateSelectedCount();
        });

        this._items.forEach(item => {
            item.checkbox?.addEventListener("change", () => {
                this.#updateSelectAllState();
                this.#updateSelectedCount();
            });
        });
    }

    // ---------- Filtering ----------

    #filter() {
        const query = this._searchQuery.toLowerCase().trim();

        this._items.forEach(item => {
            const matches = this.#matches(item, query);

            matches ? showElement(item.row) : hideElement(item.row);

            this.#updateHighlight(item, query);
        });

        this.#updateSelectAllState();
        this.#updateSelectedCount();
    }

    /**
     * Returns true if search query matches any labels/aliases
     * @param {any} item item object
     * @param {string} query query string
     * @returns {boolean} match found
     */
    #matches(item, query) {
        if (!query) return true;

        // Check all label fields (name, email, aliases column, etc.)
        const labelMatch = item.labelEls.some((/** @type {HTMLElement} */ labelEl) =>
            (labelEl.dataset.originalText || "")
                .toLowerCase()
                .includes(query)
        );

        if (labelMatch) return true;

        // Check aliases
        return item.aliases.some((/** @type {String} */ alias) =>
            alias.toLowerCase().includes(query)
        );
    }

    // ---------- Highlighting ----------

    /**
     * Highlights labels based on search query
     * @param {any} item item object
     * @param {string} query query string
     */
    #updateHighlight(item, query) {
        item.labelEls.forEach((/** @type {HTMLElement} */ labelEl) => {
            const original = labelEl.dataset.originalText || "";

            // Reset if disabled or no query
            if (!this.highlightEnabled || !query) {
                labelEl.textContent = original;
                return;
            }

            // Only highlight if THIS field matches
            if (!original.toLowerCase().includes(query)) {
                labelEl.textContent = original;
                return;
            }

            this.#highlightLabel(labelEl, query);
        });
    }

    /**
     * Highlight label based on query
     * @param {HTMLElement} element item object
     * @param {string} searchTerm query string
     */
    #highlightLabel(element, searchTerm) {
        const original = element.dataset.originalText || element.textContent || "";

        if (!element.dataset.originalText) {
            element.dataset.originalText = original;
        }

        const lowerText = original.toLowerCase();
        const lowerSearch = searchTerm.toLowerCase();

        let index = 0;
        element.textContent = "";

        while (true) {
            const matchIndex = lowerText.indexOf(lowerSearch, index);

            if (matchIndex === -1) {
                this.#addLabelPart(element, original.slice(index), false);
                break;
            }

            if (matchIndex > index) {
                this.#addLabelPart(
                    element,
                    original.slice(index, matchIndex),
                    false
                );
            }

            this.#addLabelPart(
                element,
                original.slice(matchIndex, matchIndex + searchTerm.length),
                true
            );

            index = matchIndex + searchTerm.length;
        }
    }

    /**
     * Highlight element text part
     * @param {HTMLElement} element item object
     * @param {string} part query string
     * @param {boolean} highlight whether to highlight
     */
    #addLabelPart(element, part, highlight = false) {
        const textNode = document.createTextNode(part);

        if (!highlight) {
            element.appendChild(textNode);
            return;
        }

        const span = document.createElement("span");
        span.className = "ids-highlight";
        span.appendChild(textNode);

        element.appendChild(span);
    }

    // ---------- Select-all state ----------

    #updateSelectAllState() {
        if (!this.$selectAll) return;
        const selectAll = /** @type {HTMLInputElement} */ (this.$selectAll);

        const visibleItems = this.visibleItems;

        const checkedCount = visibleItems.filter(item =>
            item.checkbox?.checked
        ).length;

        const total = visibleItems.length;

        selectAll.checked = total > 0 && checkedCount === total;
    }

    #updateSelectedCount() {
        const count = this.selectedIds.length;
        const targets = this.selectedCounters;

        targets?.forEach(el => {
            el.textContent = String(count);
            el.setAttribute("data-count", String(count));
        })
    }
}

customElements.define("ids-selectable-list", SelectableList);

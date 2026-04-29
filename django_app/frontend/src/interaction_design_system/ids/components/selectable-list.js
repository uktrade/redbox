// @ts-check

import { hideElement, isHidden, showElement } from "../../../js/utils";

export class SelectableList extends HTMLElement {
    constructor() {
        super();

        this._items = [];
        this._searchQuery = "";
    }

    static get observedAttributes() {
        return ["data-highlight"];
    }

    get visibleItems() {
        return this._items.filter(item => !isHidden(item.el));
    }

    connectedCallback() {
        this.#cacheDom();
        this.#initItems();
        this.#bindEvents();
        this.#filter(); // initial pass
    }

    attributeChangedCallback() {
        // re-run filtering to update highlighting state
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
    }

    #initItems() {
        this._items = this.$items.map(el => {
            const labelEl = el.querySelector("[data-ids-label]");
            const text = labelEl?.textContent || "";

            if (labelEl && !labelEl.dataset.originalText) {
                labelEl.dataset.originalText = text;
            }

            return {
                el,
                checkbox: el.querySelector('input[type="checkbox"]'),
                labelEl,
                label: text,
                aliases: (el.dataset.aliases || "")
                    .split(",")
                    .map(a => a.trim())
                    .filter(Boolean)
            };
        });
        console.log("items: ", this.$items, this._items);
    }

    #bindEvents() {
        this.$search?.addEventListener("input", (e) => {
            this._searchQuery = e.target.value;
            console.log("search: ", this._searchQuery);
            this.#filter();
        });

        this.$selectAll?.addEventListener("change", (evt) => {
            const checked = evt.target.checked;

            this.visibleItems.forEach(item => {
                if (item.checkbox) {
                    item.checkbox.checked = checked;
                }
            });

        });

        this._items.forEach(item => {
            item.checkbox?.addEventListener("change", () => {
                this.#updateSelectAllState();
            });
        });
    }

    // ---------- Filtering ----------

    #filter() {
        const query = this._searchQuery.toLowerCase().trim();

        this._items.forEach(item => {
            const matches = this.#matches(item, query);
            matches ? showElement(item.el) : hideElement(item.el);

            this.#updateHighlight(item, query);
        });

        this.#updateSelectAllState();
    }

    #matches(item, query) {
        if (!query) return true;

        if (item.label.toLowerCase().includes(query)) return true;

        return item.aliases.some(alias =>
            alias.toLowerCase().includes(query)
        );
    }

    // ---------- Highlighting ----------

    #updateHighlight(item, query) {
        if (!item.labelEl) return;

        const original = item.labelEl.dataset.originalText || "";

        // reset if disabled or no query
        if (!this.highlightEnabled || !query) {
            item.labelEl.textContent = original;
            return;
        }

        this.#highlightLabel(item.labelEl, query);
    }

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
                this.#addLabelPart(element, original.slice(index, matchIndex), false);
            }

            this.#addLabelPart(
                element,
                original.slice(matchIndex, matchIndex + searchTerm.length),
                true
            );

            index = matchIndex + searchTerm.length;
        }
    }

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

        const visibleItems = this.visibleItems;

        const checkedCount = visibleItems.filter(item =>
            item.checkbox?.checked
        ).length;

        const total = visibleItems.length;

        this.$selectAll.checked = total > 0 && checkedCount === total;
    }
}

customElements.define("ids-selectable-list", SelectableList);

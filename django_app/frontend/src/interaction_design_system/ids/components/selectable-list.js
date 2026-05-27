// @ts-check

export class SelectableList extends HTMLElement {

    constructor() {
        super();
        this.selectedIds = new Set();
    }


    connectedCallback() {
        this.cacheDom();
        this.bindEvents();
        this.restoreSelections();
        this.updateSelectedCount();
        this.updateSelectAllState();
    }


    get rows() {
        return /** @type {NodeListOf<HTMLElement>} */ (
            this.querySelectorAll("[data-user-row]")
        );
    }


    /**
     * DOM update hook (HTMX)
     */
    onDomUpdated() {
        this.restoreSelections();
        this.updateSelectAllState();
        this.updateSelectedCount();
    }


    cacheDom() {
        this.selectedCountEls = /** @type {NodeListOf<HTMLElement>} */ (
            this.querySelectorAll("[data-selected-count]")
        );
        this.selectAll = /** @type {HTMLInputElement} */ (
            this.querySelector("[data-select-all]")
        );
        this.selectedInputsContainer = /** @type {HTMLElement} */ (
            this.querySelector("[data-selected-inputs]")
        );
    }


    bindEvents() {

        // Individual checkbox changes
        this.addEventListener("change", (event) => {
            const checkbox = event.target;

            if (!(checkbox instanceof HTMLInputElement)) return;
            if (checkbox.type !== "checkbox") return;

            const row = /** @type {HTMLElement} */ (
                checkbox.closest("[data-user-row]")
            );
            if (!row) return;

            const userId = row.dataset.userId;
            if (!userId) return;

            if (checkbox.checked) {
                this.selectedIds.add(userId);
            } else {
                this.selectedIds.delete(userId);
            }

            this.syncHiddenInputs();
            this.updateSelectedCount();
            this.updateSelectAllState();
        });

        // HTMX swap hook
        document.body.addEventListener("htmx:afterSwap", (evt) => {
            const detail = /** @type {any} */ (evt).detail;

            if (!detail?.target) return;
            if (!this.contains(detail.target)) return;

            this.onDomUpdated();
        });

        // Select all
        this.selectAll?.addEventListener("change", (evt) => {
            const target = /** @type {HTMLInputElement} */ (evt.target);
            const checked = target.checked;

            this.rows?.forEach(row => {
                const checkbox = /** @type {HTMLInputElement} */ (
                    row.querySelector('input[type="checkbox"]')
                );

                const userId = row.dataset.userId;

                if (!checkbox || !userId) return;

                checkbox.checked = checked;

                if (checked) {
                    this.selectedIds.add(userId);
                } else {
                    this.selectedIds.delete(userId);
                }
            });

            this.syncHiddenInputs();
            this.updateSelectedCount();
            this.updateSelectAllState();
        });
    }


    /**
     * Restore state after HTMX swaps
     */
    restoreSelections() {
        this.rows?.forEach(row => {
            const userId = row.dataset.userId;
            const checkbox = /** @type {HTMLInputElement} */ (
                row.querySelector('input[type="checkbox"]')
            );

            if (!checkbox || !userId) return;

            checkbox.checked = this.selectedIds.has(userId);
        });
    }


    updateSelectAllState() {
        const selectAll = this.selectAll;
        if (!selectAll) return;

        let checked = 0;
        let total = 0;

        this.rows?.forEach(row => {
            const checkbox = row.querySelector('input[type="checkbox"]');
            const userId = row.dataset.userId;

            if (!checkbox || !userId) return;

            total++;

            if (this.selectedIds.has(userId)) {
                checked++;
            }
        });

        selectAll.checked = total > 0 && checked === total;
        selectAll.indeterminate = checked > 0 && checked < total;
    }


    syncHiddenInputs() {
        if (!this.selectedInputsContainer) return;

        this.selectedInputsContainer.innerHTML = "";

        this.selectedIds.forEach(userId => {
            const input = document.createElement("input");
            input.type = "hidden";
            input.name = "user_ids";
            input.value = userId;

            this.selectedInputsContainer?.appendChild(input);
        });
    }


    updateSelectedCount() {
        const count = this.selectedIds.size;

        this.selectedCountEls?.forEach(el => {
            el.textContent = String(count);
        });
    }
}

customElements.define("ids-selectable-list", SelectableList);

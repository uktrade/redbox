// @ts-check

export class SidePanel extends HTMLElement {
    collapsedClass = "rbds-side-panel-wrapper--collapsed";
    storageKey = "rbds-side-panel-collapsed";

    constructor() {
        super();

        // Apply persisted state from localStorage
        const persisted = localStorage.getItem(this.storageKey) === "true";
        if (persisted) {
            this.classList.add(this.collapsedClass);
        }

        // Set aria-expanded
        this.setAttribute('aria-expanded', this.expanded ? 'true' : 'false');
    }

    connectedCallback() {
        // Bind toggle buttons
        this.toggle.forEach((element) => {
            element.addEventListener("click", () => this.togglePanel());
        });

        // console.log("SidePanel initialized", this.toggle);
    }

    get toggle() {
        return this.querySelectorAll('[slot="toggle"]');
    }

    get expanded() {
        // True if NOT collapsed
        return !this.classList.contains(this.collapsedClass);
    }

    togglePanel() {
        this.classList.toggle(this.collapsedClass);

        // Update aria
        this.setAttribute('aria-expanded', this.expanded ? 'true' : 'false');

        // Persist state
        localStorage.setItem(this.storageKey, String(this.classList.contains(this.collapsedClass)));

        // Update cookie for server-side persistence
        document.cookie = `${this.storageKey}=${!this.expanded}; path=/`;
    }
}

customElements.define("rbds-side-panel", SidePanel);

// @ts-check

import { toggleNoScroll } from "./no-scroll";

export class SidePanel extends HTMLElement {
    collapsedClass = "rbds-side-panel-wrapper--collapsed";
    storageKey = "rbds-side-panel-collapsed";

    constructor() {
        super();

        // Apply persisted state from localStorage
        const collapsed = localStorage.getItem(this.storageKey) === "true";
        if (collapsed) this.classList.add(this.collapsedClass);

        // Apply no-scroll based on persisted state
        this.sidepanelId = toggleNoScroll(!collapsed, this.sidepanelId);

        // Set aria-expanded
        this.setAttribute('aria-expanded', this.expanded ? 'true' : 'false');
    }


    connectedCallback() {
        // Bind toggle buttons
        this.toggle.forEach((element) => {
            element.addEventListener("click", () => this.togglePanel());
        });
    }


    /**
     * Returns the sidepanel ID
     * @returns {String | null | undefined} sidepanel ID
     */
    get sidepanelId () {
        return this._sidepanelId;
    }


    /**
     * Returns the sidepanel ID
     * @param {String | null} id sidepanel ID
     */
    set sidepanelId (id) {
        this._sidepanelId = id;
    }


    get toggle() {
        return document.querySelectorAll('[slot="toggle-side-panel"]');
    }


    get expanded() {
        // True if NOT collapsed
        return !this.classList.contains(this.collapsedClass);
    }


    togglePanel() {
        this.classList.toggle(this.collapsedClass);

        // Update scroll
        this.sidepanelId = toggleNoScroll(this.expanded, this.sidepanelId);

        // Update aria
        this.setAttribute('aria-expanded', this.expanded ? 'true' : 'false');

        // Persist state
        localStorage.setItem(this.storageKey, String(this.classList.contains(this.collapsedClass)));

        // Update cookie for server-side persistence
        document.cookie = `${this.storageKey}=${!this.expanded}; path=/`;
    }
}
customElements.define("rbds-side-panel", SidePanel);

// @ts-check

import { getBreakpointPx } from "../utils";
import { disableNoScroll, enableNoScroll } from "./no-scroll";

export class SidePanel extends HTMLElement {
    storageKey = "rbds-side-panel-collapsed";
    collapsedClass = "rbds-side-panel-wrapper--collapsed";
    noScrollSource = "side-panel";
    toggleSlot = "toggle-side-panel";
    maxOverlap = getBreakpointPx("m");
    mediaQuery = window.matchMedia(`(min-width: ${this.maxOverlap}px)`);

    constructor() {
        super();

        const collapsed = localStorage.getItem(this.storageKey) === "true";
        collapsed ? this.close() : this.open();
    }


    connectedCallback() {
        // Bind toggle buttons
        this.toggleElements.forEach((element) => {
            element.addEventListener("click", (evt) => {
                evt.stopPropagation();
                this.togglePanel();
            });
        });
        // Handle screen size changes
        this.mediaQuery.addEventListener("change", this.handleScreenSizeChanges);
    }


    handleScreenSizeChanges= () => {
        if (!this.expanded || this.noOverlap) {
            document.removeEventListener("click", this.handleOutsideClick);
        } else {
            document.addEventListener("click", this.handleOutsideClick);
        }
    }


    handleOutsideClick = (/** @type {Event} */ evt) => {
        const target = /** @type {HTMLElement} */ (evt.target);
        const link = target.closest('a[href]')?.getAttribute('href') ? true : false;

        // Click inside side-panel (excluding links) - ignore
        if (this.contains(target) && !link) return;

        // Click on toggle - ignore
        if (target?.closest(`[slot="${this.toggleSlot}"]`)) return;

        this.close();
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


    get toggleElements() {
        return document.querySelectorAll(`[slot="${this.toggleSlot}"]`);
    }


    get expanded() {
        // True if NOT collapsed
        return !this.classList.contains(this.collapsedClass);
    }


    get noOverlap() {
        return this.mediaQuery.matches;
    }


    togglePanel() {
        this.expanded ? this.close() : this.open();
    }


    open() {
        this.classList.remove(this.collapsedClass);

        // Enable no-scroll
        this.sidepanelId = enableNoScroll(this.noScrollSource);

        // Update aria
        this.setAttribute('aria-expanded', 'true');

        // Persist state
        localStorage.setItem(this.storageKey, "false");

        // Update cookie for server-side persistence
        document.cookie = `${this.storageKey}=false; path=/`;

        if (!this.noOverlap) document.addEventListener("click", this.handleOutsideClick);
    }


    close() {
        this.classList.add(this.collapsedClass);

        // Disable no-scroll
        if (this.sidepanelId) disableNoScroll(this.sidepanelId);

        // Update aria
        this.setAttribute('aria-expanded', 'false');

        // Persist state
        localStorage.setItem(this.storageKey, "true");

        // Update cookie for server-side persistence
        document.cookie = `${this.storageKey}=true; path=/`;

        document.removeEventListener("click", this.handleOutsideClick);
    }
}
customElements.define("rbds-side-panel", SidePanel);

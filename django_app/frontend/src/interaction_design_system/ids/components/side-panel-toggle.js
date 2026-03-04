// @ts-check

export class SidePanelToggle extends HTMLElement {
    toggleEventId = "side-panel-toggle";

    connectedCallback() {
        const toggleEvent = new CustomEvent(this.toggleEventId, {
            detail: this,
        });

        this.toggleElement?.addEventListener("click", (evt) => {
            document.dispatchEvent(toggleEvent);
        });
    }


    /**
     * Returns the nested toggle click element
     * @returns { Element } toggle element
     */
    get toggleElement () {
        return this.children[0];
    }
}
customElements.define("ids-side-panel-toggle", SidePanelToggle);

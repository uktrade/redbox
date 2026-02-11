// @ts-check

export class SidePanelToggle extends HTMLElement {
    toggleEventId = "side-panel-toggle";

    connectedCallback() {
        console.log("SidePanelToggle connectedCallback", this.toggleElement);
        const toggleEvent = new CustomEvent(this.toggleEventId, {
            detail: this,
        });

        this.toggleElement?.addEventListener("click", (evt) => {
            console.log("Fired toggle event");
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
customElements.define("rbds-side-panel-toggle", SidePanelToggle);

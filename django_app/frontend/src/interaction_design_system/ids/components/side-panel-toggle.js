// @ts-check

import { emitEvent, Events } from "../events";

export class SidePanelToggle extends HTMLElement {
    connectedCallback() {
        this.toggleElement?.addEventListener("click", () => {
            emitEvent(Events.SIDE_PANEL_TOGGLE, {SidePanelToggle:this});
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

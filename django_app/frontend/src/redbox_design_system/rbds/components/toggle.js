// @ts-check

export class Toggle extends HTMLElement {
    constructor() {
        super();
        this.state = this.getAttribute("data-state") || "off";
    }

    connectedCallback() {
        this.setAttribute("data-state", this.state);

        this.addEventListener("click", () => this.toggle());
        this.setAttribute("role", "button");
        this.setAttribute("tabindex", "0");
    }

    toggle() {
        this.state = this.state === "on" ? "off" : "on";
        this.setAttribute("data-state", this.state);

        this.dispatchEvent(
            new CustomEvent("rbds:toggle", {
                bubbles: true,
                detail: { state: this.state },
            })
        );
    }
}
customElements.define("rbds-toggle", Toggle);

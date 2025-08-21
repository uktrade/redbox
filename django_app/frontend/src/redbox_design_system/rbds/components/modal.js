// @ts-check

export class Modal extends HTMLElement {

    connectedCallback() {
        this.#initialiseModal();
    }

    get dialog() {
        return this.querySelector("dialog");
    }


    #initialiseModal() {
        if (!this.dialog) return;

        const openElement = this.querySelector(".modal-open");
        const closeElements = this.querySelectorAll(".modal-close");

        openElement?.addEventListener("click", (evt) => {
            evt.preventDefault();
            if (!this.dialog) return;

            this.dialog.showModal();
            document.documentElement.classList.add("no-scroll");
        });

        closeElements.forEach((element) => {
            element.addEventListener("click", (evt) => {
                evt.preventDefault();
                if (!this.dialog) return;

                this.dialog.close();
                document.documentElement.classList.remove("no-scroll");
            });
        });

        this.dialog.addEventListener("click", function (event) {
            const rect = this.getBoundingClientRect();
            const isInDialog =
                rect.top <= event.clientY &&
                event.clientY <= rect.top + rect.height &&
                rect.left <= event.clientX &&
                event.clientX <= rect.left + rect.width;
            if (!isInDialog) {
                this.close();
                document.documentElement.classList.remove("no-scroll");
            }
        });
    }
}
customElements.define("rbds-modal", Modal);

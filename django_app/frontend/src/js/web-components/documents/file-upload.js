// @ts-check

class FileUpload extends HTMLElement {
    connectedCallback() {
        this.querySelector("button")?.addEventListener("click", (evt) => {
            evt.preventDefault();
            this.querySelector("input")?.click();
        });
    }
}
customElements.define("file-upload", FileUpload);

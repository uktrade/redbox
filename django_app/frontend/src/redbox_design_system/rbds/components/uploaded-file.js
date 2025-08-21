// @ts-check

export class UploadedFile extends HTMLElement {
    constructor() {
        super();

        this.templateId = 'uploaded-file-template';
        this.template = /** @type {HTMLTemplateElement} */ (document.getElementById(this.templateId));
    }

    static StatusTypes = Object.freeze({
        COMPLETE: "complete",
        PROCESSING: "processing",
        ERROR: "error",
    });

    static FileStatusDisplay = Object.freeze({
        [UploadedFile.StatusTypes.COMPLETE]: "Ready to use",
        [UploadedFile.StatusTypes.PROCESSING]: "Processing",
        [UploadedFile.StatusTypes.ERROR]: "Error",
    });


    connectedCallback() {
        if (!this.template) return console.warn(`UploadedFile: Template '${this.templateId}' not found`);

        const uploadedFileWrapper = /** @type {HTMLDivElement} */ (
            this.template.content.firstElementChild?.cloneNode(true)
        );
        this.appendChild(uploadedFileWrapper);
    }


    /**
     * Returns the upload file ID
     * @returns {String | undefined} File UUID
     */
    get fileId() {
        return this._fileId;
    }


    /**
     * Sets the id for the file
     * @param {String} value UUID from the server
     */
    set fileId(value) {
        this._fileId = value;
        this.dataset.id = value;
    }


    /**
     * Returns the upload status for the file
     * @returns {String} uploaded status
     */
    get status() {
        return this._status || UploadedFile.StatusTypes.PROCESSING;
    }


    /**
     * Sets the upload status for the file
     * @param {String} value status from server
     */
    set status(value) {
        this._status = value.toLowerCase();
        this.statusTextElement.innerText = this.#getDisplayStatus();
        this.progressBarElement.dataset.status = this.status;
        if (this.status === UploadedFile.StatusTypes.COMPLETE) this.progressPercent = 100;
    }


    /**
     * Returns the name of the file
     * @returns {String | undefined} File name
     */
    get fileName() {
        return this._fileName;
    }


    /**
     * Sets the name of the uploaded file and updates the icon
     * @param {String} value File name
     */
    set fileName(value) {
        this._fileName = value;
        this.nameElement.innerText = value;
        this.nameElement.title = value;
        this.#updateIconDisplay();
    }


    /**
     * Returns the upload progress percentage
     * @returns {Number} Upload progress percentage
     */
    get progressPercent() {
        return this._progressPercent || 0;
    }


    /**
     * Sets the upload progress percentage
     * @param {Number} percent Upload progress percentage
     */
    set progressPercent(percent) {
        this._progressPercent = percent;
        this.progressPercentElement.innerText = percent.toString();
        this.progressBarElement.style.setProperty("--progress-width", `${percent}%`);
    }

    get iconElement() {
        return /** @type {HTMLElement} */ (this.querySelector('[data-icon]'));
    }


    get nameElement() {
        return /** @type {HTMLElement} */ (this.querySelector('[data-name]'));
    }


    get statusTextElement() {
        return /** @type {HTMLElement} */ (this.querySelector('[data-status-text]'));
    }


    get progressBarElement() {
        return /** @type {HTMLElement} */ (this.querySelector('[data-progress-bar]'));
    }


    get progressPercentElement() {
        return /** @type {HTMLElement} */ (this.querySelector('[data-progress-percent]'));
    }


    get removeElement() {
        return /** @type {HTMLElement} */ (this.querySelector('[data-remove]'));
    }


    /**
     * Returns the display text for the file upload status from the server
     * @param {String} status - Status from the server
     * @returns {String} Display text
     */
    #getDisplayStatus(status = this.status) {
        return UploadedFile.FileStatusDisplay[status.toLowerCase()] || status;
    }


    /**
     * Updates the icon display element with the correct icon for the file type
     */
    #updateIconDisplay() {
        const fileExt = this.fileName?.split(".")[1];
        console.log("TODO - Implement Icon: ", fileExt, this.iconElement);
    }
}
customElements.define("uploaded-file", UploadedFile);

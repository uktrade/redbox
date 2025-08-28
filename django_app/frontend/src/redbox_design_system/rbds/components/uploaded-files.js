// @ts-check

import { UploadedFile } from "./uploaded-file";

export class UploadedFiles extends HTMLElement {
    constructor() {
        super();

        this.templateId = 'uploaded-files-template';
        this.template = /** @type {HTMLTemplateElement} */ (document.getElementById(this.templateId));

        /** @type {UploadedFile[]} */
        this.files = [];
    }


    connectedCallback() {
        if (!this.template) return console.warn(`UploadedFiles: Template '${this.templateId}' not found`);

        const uploadedFilesWrapper = /** @type {HTMLDivElement} */ (
            this.template.content.firstElementChild?.cloneNode(true)
        );
        this.appendChild(uploadedFilesWrapper);
        this.setAttribute("contenteditable", "false");
        this.#monitorProcessingStatus();
    }


    get container() {
        return /** @type {HTMLElement} */ (this.querySelector('#uploaded-files'));
    }


    /**
     * Adds a new file to the list
     * @param {String} name - File name
     * @returns {UploadedFile} UploadedFile element
     */
    addFile(name) {
        const uploadedFile = /** @type {UploadedFile} */ (document.createElement("uploaded-file"));
        this.container.appendChild(uploadedFile);
        uploadedFile.fileName = name;
        this.files.push(uploadedFile);

        uploadedFile.removeElement.addEventListener("click", (evt) => {
            evt.stopPropagation();
            this.removeFile(uploadedFile);
        });

        return uploadedFile;
    }


    /**
     * Remove a file
     * @param {UploadedFile} uploadedFile - File element
     * @returns {boolean} Removal success flag
     */
    removeFile(uploadedFile) {
        const index = this.files?.indexOf(uploadedFile);
        if (index === -1 || index === undefined) return false; // File not found
        this.files.splice(index, 1); // Remove file from array
        if (uploadedFile.parentNode === this.container) this.container.removeChild(uploadedFile);
        if (this.allProcessed()) this.#emitProcessedEvent();
        if (this.isEmpty()) this.remove();
        return true;
    }


    /**
     * Retrieves the file element by ID
     * @param {String} id - File string UUID
     * @returns {UploadedFile | null} UploadedFile element
     */
    getFileById(id) {
        return this.files.find(file => file.dataset.id === id) || null;
    }


    /**
     * Checks whether all files have finished processing
     * @returns {Boolean} Uploaded status of all files
     */
    allProcessed() {
        if (this.files.length === 0) return true;

        return this.files.every(file =>
            [
                UploadedFile.StatusTypes.COMPLETE.valueOf(),
                UploadedFile.StatusTypes.ERROR.valueOf(),
            ].includes(file.status)
        );
    }


    /**
     * Checks if the files lsit is empty
     * @returns {boolean} Flag indicating if files array is empty
     */
    isEmpty() {
        return (this.files.length === 0);
    }

    /**
     * Emits an event to signify that all uploads have finished processing
     */
    #emitProcessedEvent() {
        document.body.dispatchEvent(new CustomEvent("file-uploads-processed"));
    }

    /**
     * Monitors the processing status of each uploaded file
     */
    #monitorProcessingStatus() {
        document.body.addEventListener("file-upload-processed", (evt) => {
            const uploadedFile = /** @type {CustomEvent} */ (evt).detail;
            if (uploadedFile.parentNode !== this.container) return;
            if (this.allProcessed()) this.#emitProcessedEvent();
        });
    }
}
customElements.define("uploaded-files", UploadedFiles);

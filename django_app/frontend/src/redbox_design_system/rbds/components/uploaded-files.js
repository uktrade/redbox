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
     * Checks whether all files have finished uploading
     * @returns {Boolean} Uploaded status of all files
     */
    allCompleted() {
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
}
customElements.define("uploaded-files", UploadedFiles);

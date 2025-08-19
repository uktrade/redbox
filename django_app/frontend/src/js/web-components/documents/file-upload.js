// @ts-check

import { pollFileStatus, updateYourDocuments } from "../../services";

class FileUpload extends HTMLElement {
    connectedCallback() {
        this.#bindFileInputEvent();
        this.#bindTextboxEvents();
    }


    /**
     * Textarea for uploading files via drag/drop
    */
    get textarea() {
        const textareaSelector = this.getAttribute("textarea-selector");
        if (!textareaSelector) return null;

        return /** @type {HTMLElement} */ (document.querySelector(textareaSelector));
    }


    /**
     * File input element for triggering the file picker
    */
    get fileInput() {
        const fileInputSelector = this.getAttribute("file-input-selector");
        if (!fileInputSelector) return null;

        return /** @type {HTMLInputElement} */ (document.querySelector(fileInputSelector));
    }


    /**
     * Custom element for triggering input file element
    */
    get fileUploadTrigger() {
        const fileUploadTriggerSelector = this.getAttribute("file-upload-trigger-selector");
        if (!fileUploadTriggerSelector) return null;

        return /** @type {HTMLElement} */ (document.querySelector(fileUploadTriggerSelector));
    }


    /**
     * URL for file uploads
    */
    get uploadUrl() {
        return this.getAttribute("upload-url");
    }


    /**
     * Form element for submission
    */
    get form() {
        const formSelector = this.getAttribute("form-selector");
        if (!formSelector) return null;

        return /** @type {HTMLFormElement} */ (document.querySelector(formSelector));
    }


    /**
     * Container for file upload textbox UI elements
    */
    get uploadedFilesWrapper() {
        return /** @type {HTMLDivElement} */ (this.textarea?.querySelector(".uploaded-files-wrapper"));
    }


    /**
     * Allow a custom element to trigger the file upload input element
     * @param {HTMLInputElement | null} fileInput - File upload input element
     * @param {HTMLElement | null} fileUploadTrigger - Custom trigger element
     */
    #bindFileInputEvent(fileInput = this.fileInput, fileUploadTrigger = this.fileUploadTrigger) {
        if (!fileInput || !fileUploadTrigger) return;

        fileUploadTrigger.addEventListener("click", (evt) => {
            evt.preventDefault();
            this.fileInput?.click();
        });
    }


    /**
     * Binds file upload events for a textarea is provided
     */
    #bindTextboxEvents(textarea = this.textarea) {
        if (!textarea) return;

        // Attach drag/drop upload functionality to textbox
        if (textarea.classList.contains("drop-zone")) {
            textarea.addEventListener("dragover", (evt) => {
                evt.preventDefault();
                this.textarea?.classList.add("drag-over");
            });

            textarea.addEventListener("dragleave", (evt) => {
                this.textarea?.classList.remove("drag-over");
            });

            textarea.addEventListener("drop", (evt) => {
                evt.preventDefault();
                this.textarea?.classList.remove("drag-over");
                const files = evt.dataTransfer?.files;
                this.#handleFiles(files);
            });
        }

        // Remove uploaded files UI elements before submission
        this.form?.addEventListener("submit", (evt) => {
            evt.preventDefault();
            this.uploadedFilesWrapper?.remove();
            if (this.textarea?.textContent?.trim()) this.form?.submit();
        });

        if (this.fileInput) {
            this.fileInput.addEventListener("change", (evt) => {
                const target = /** @type {HTMLInputElement} */ (evt.target);
                this.#handleFiles(target.files);
                target.value = "";
            });
        }

        document.body.addEventListener("doc-complete", (evt) => {
            const id = /** @type {CustomEvent} */ (evt).detail.id;
            if (!id) return;

            const uploadedFileWrapper = /** @type {HTMLDivElement} */ (
                document.querySelector(`[data-id="${id}"]`)
            );
            if (!uploadedFileWrapper) return;

            const uploadProgressText = /** @type {HTMLDivElement} */ (
                uploadedFileWrapper.querySelector(".upload-progress-text")
            );
            const uploadProgressFill = /** @type {HTMLDivElement} */ (
                uploadedFileWrapper.querySelector(".upload-progress-fill")
            );

            uploadProgressFill.dataset.status = "complete";
            uploadProgressText.textContent = "Ready to use";

            updateYourDocuments().then(() => { this.#checkDocument(id) });
        });

        document.body.addEventListener("doc-error", (evt) => {
            const id = /** @type {CustomEvent} */ (evt).detail.id;
            if (!id) return;

            const uploadedFileWrapper = /** @type {HTMLDivElement} */ (
                this.querySelector(`[data-id="${id}"]`)
            );
            if (!uploadedFileWrapper) return;

            const uploadProgressText = /** @type {HTMLDivElement} */ (
                uploadedFileWrapper.querySelector(".upload-progress-text")
            );
            const uploadProgressFill = /** @type {HTMLDivElement} */ (
                uploadedFileWrapper.querySelector(".upload-progress-fill")
            );

            uploadProgressFill.dataset.status = "error";
            uploadProgressText.textContent = "Error";
        });
    }


    /**
     * Process files for upload
     * @param {FileList | null | undefined} files - File(s) to be uploaded
     */
    #handleFiles(files) {
        if (!files) return;
        const moveCaretToEnd = (!this.textarea?.textContent?.trim());
        for (const file of files) {
            const uploadedFileElement = this.#addFileToTextBox(file);
            this.uploadFile(file, uploadedFileElement);
        }
        if (moveCaretToEnd) {
            this.textarea?.appendChild(document.createTextNode("\n"));
            this.#moveCaretToEnd();
        }
    }


    /**
     * Add file display UI elements to textbox
     * @param {File} file - File to be uploaded
     */
    #addFileToTextBox(file, textarea = this.textarea) {
        if (!textarea) return;

        let uploadedFiles = /** @type {HTMLDivElement} */ (textarea.querySelector(".uploaded-files"));

        if (!this.uploadedFilesWrapper) {
            const template = /** @type {HTMLTemplateElement} */ (document.getElementById('uploaded-files-template'));
            const uploadedFilesWrapper = /** @type {HTMLDivElement} */ (template?.content.firstElementChild?.cloneNode(true));
            uploadedFiles = /** @type {HTMLDivElement} */ (uploadedFilesWrapper.querySelector(".uploaded-files"));
            textarea.prepend(uploadedFilesWrapper);
        }
        let uploadedFileElement = this.#createFileElement(file);
        uploadedFiles.appendChild(uploadedFileElement);
        return uploadedFileElement;
    }


    /**
     * Creates a UI element for an uploaded file to be added to the textarea
     * @param {File} file - File to be uploaded
     */
    #createFileElement(file) {
        const template = /** @type {HTMLTemplateElement} */ (document.getElementById('uploaded-file-template'));
        const uploadedFileWrapper = /** @type {HTMLDivElement} */ (template?.content.firstElementChild?.cloneNode(true));

        const name = /** @type {HTMLDivElement} */ (uploadedFileWrapper.querySelector(".file-name"));
        name.textContent = file.name;
        name.setAttribute("title", file.name);

        // TODO: Replace with proper icons
        const icon = /** @type {HTMLDivElement} */ (uploadedFileWrapper.querySelector(".file-icon"));
        icon.textContent = "📁";

        const removeLink = /** @type {HTMLAnchorElement} */ (uploadedFileWrapper.querySelector(".action-link"));
        removeLink.addEventListener("click", (evt) => {
            evt.stopPropagation();
            uploadedFileWrapper.remove();
            const uploadedFiles = this.textarea?.querySelector(".uploaded-files");
            const uploadedFilesWrapper = this.textarea?.querySelector(".uploaded-files-wrapper");
            if (uploadedFiles?.innerHTML === "") uploadedFilesWrapper?.remove();
            updateYourDocuments().then(() => { this.#checkDocument(uploadedFileWrapper.dataset.id, false) });
        });
        return uploadedFileWrapper;
    }


    /**
     * Triggers file upload to the server with the provided url
     * @param {File} file - File to be uploaded
     * @param {HTMLElement | undefined} uploadedFileElement - UI element for uploaded file
     */
    uploadFile(file, uploadedFileElement, uploadUrl = this.uploadUrl) {
        if (!file || !uploadedFileElement || !uploadUrl) return;

        const progressFill = /** @type {HTMLDivElement} */ (uploadedFileElement.querySelector(".upload-progress-fill"));
        const progressFillText = /** @type {HTMLDivElement} */ (uploadedFileElement.querySelector(".upload-progress-fill-text"));
        const progressText = /** @type {HTMLDivElement} */ (uploadedFileElement.querySelector(".upload-progress-text"));

        const csrfToken = /** @type {HTMLInputElement | null} */ (
            document.querySelector('[name="csrfmiddlewaretoken"]')
        )?.value || "";

        const xhr = new XMLHttpRequest();
        xhr.open("POST", uploadUrl);
        xhr.setRequestHeader("X-CSRFToken", csrfToken);

        xhr.upload.addEventListener("progress", (evt) => {
            if (evt.lengthComputable) {
                const percent = (evt.loaded / evt.total) * 100;
                progressFill.style.setProperty("--progress-width", `${percent}%`);
                progressFillText.innerText = percent.toString();
            }
        });

        xhr.onload = () => {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);

                // TODO: Define status types somewhere?
                if (response.errors || response.status in ["errored", "deleted"]) {
                    progressFill.dataset.status = "error";
                    progressText.innerText = "Error";
                    console.error(response.errors);
                }
                if (response.file_id) uploadedFileElement.dataset.id = response.file_id;

                if (response.file_name !== file.name) {
                    const fileNameDiv = /** @type {HTMLDivElement} */ (uploadedFileElement.querySelector(".file-name"));
                    fileNameDiv.textContent = response.file_name;
                    fileNameDiv.setAttribute("title", response.file_name);
                }

                if (response.status === "complete") {
                    progressFill.dataset.status = "complete";
                    progressText.innerText = "Ready to use";
                    updateYourDocuments().then(() => { this.#checkDocument(response.file_id) });
                }
                progressFill.style.setProperty("--progress-width", "100%");
                progressFillText.innerText = "100";

                if (response.status === "processing") pollFileStatus(response.file_id);
            } else {
                progressFill.dataset.status = "error";
                progressText.innerText = "Error";
            }
        };

        xhr.onerror = () => {
            progressFill.dataset.status = "error";
            progressText.innerText = "Error";
        };

        const formData = new FormData();
        formData.append("file", file);
        xhr.send(formData);
    }


    /**
     * Check/uncheck the specified document
     * @param {string | null | undefined} id - Document ID
     * @param {boolean} checked - checked value
    */
    #checkDocument(id, checked = true) {
        if (!id) return;

        const checkbox = /** @type {HTMLInputElement} */ (document.querySelector(`#file-${id}`));
        if (!checkbox) return;
        checkbox.checked = checked;
    }


    /**
     * Moves the caret to the end of the textarea
    */
    #moveCaretToEnd(textarea = this.textarea) {
        if (!textarea) return;

        textarea.focus();
        const range = document.createRange();
        range.selectNodeContents(textarea);
        range.collapse(false);
        const selection = window.getSelection();
        selection?.removeAllRanges();
        selection?.addRange(range);
    }
}
customElements.define("file-upload", FileUpload);

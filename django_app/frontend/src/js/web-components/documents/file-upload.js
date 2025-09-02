// @ts-check

import { UploadedFiles, UploadedFile } from "../../../redbox_design_system/rbds/components";
import { pollFileStatus, updateYourDocuments } from "../../services";
import { getCsrfToken } from "../../utils";
import { MessageInput } from "../chats/message-input";

class FileUpload extends HTMLElement {
    constructor() {
        super();
        this.uploadedFileIds = new Set();
        this.deselectedFileIds = new Set();
    }

    connectedCallback() {
        this.input = /** @type {HTMLInputElement} */ (this.querySelector('input[type="file"]'));
        this.button = /** @type {HTMLButtonElement} */ (this.querySelector('button'));

        this.#bindButtonInputEvent();
        this.#bindTextboxEvents();
    }


    /**
     * Textarea for uploading files via drag/drop
    */
    get textarea() {
        return this.messageInput.textarea;
    }


    /**
     * URL for file uploads
    */
    get uploadUrl() {
        return this.getAttribute("upload-url");
    }


    /**
     * Container for file upload textbox UI elements
    */
    get uploadedFiles() {
        return /** @type {UploadedFiles} */ (this.textarea?.querySelector("uploaded-files"));
    }


    /**
     * Message Input component
    */
    get messageInput() {
        if (!this._messageInput || !document.body.contains(this._messageInput)) {
            this._messageInput = /** @type {MessageInput} */ (
                document.querySelector('message-input')
            );
        }
        return this._messageInput;
    }


    /**
     * Setter for uploadedFiles object
     * @param {String | null | undefined} id - File UUID from the server
    */
    addSelectedFileId(id) {
        if (!id) return;
        this.uploadedFileIds.add(id);
        this.deselectedFileIds.delete(id);
    }


    /**
     * Removes a file from the uploadedFiles object
     * @param {String | null | undefined} id - File UUID from the server
    */
    removeSelectedFileId(id) {
        if (!id) return;
        this.uploadedFileIds.delete(id);
        this.deselectedFileIds.add(id);
    }


    /**
     * Disables the send/dictate functionality for the chat
    */
    #disableSubmit() {
        if (this.messageInput) this.messageInput.disableSubmit();
    }


    /**
     * Enables the send/dictate functionality for the chat
    */
    #enableSubmit() {
        if (this.messageInput) this.messageInput.enableSubmit();
    }


    /**
     * Allow a custom element to trigger the file upload input element
     */
    #bindButtonInputEvent() {
        this.button?.addEventListener("click", (evt) => {
            evt.preventDefault();
            this.input?.click();
        });
    }


    /**
     * Binds file upload events for a textarea is provided
     */
    #bindTextboxEvents() {
        if (!this.textarea) return;

        // Attach drag/drop upload functionality to textbox
        if (this.textarea.classList.contains("drop-zone")) {
            this.textarea.addEventListener("dragover", (evt) => {
                evt.preventDefault();
                this.textarea?.classList.add("drag-over");
            });

            this.textarea.addEventListener("dragleave", (evt) => {
                this.textarea?.classList.remove("drag-over");
            });

            this.textarea.addEventListener("drop", (evt) => {
                evt.preventDefault();
                this.textarea?.classList.remove("drag-over");
                const files = evt.dataTransfer?.files;
                this.#handleFiles(files);
            });
        }

        this.input?.addEventListener("change", (evt) => {
            const target = /** @type {HTMLInputElement} */ (evt.target);
            this.#handleFiles(target.files);
            target.value = "";
        });

        document.body.addEventListener("doc-complete", (evt) => {
            const id = /** @type {CustomEvent} */ (evt).detail.id;
            if (!id) return;

            const uploadedFile = this.uploadedFiles.getFileById(id);
            if (!uploadedFile) return;

            uploadedFile.status = UploadedFile.StatusTypes.COMPLETE;

            updateYourDocuments().finally(() => this.#checkDocuments(id));
        });

        document.body.addEventListener("doc-error", (evt) => {
            const id = /** @type {CustomEvent} */ (evt).detail.id;
            if (!id) return;

            const uploadedFile = this.uploadedFiles.getFileById(id);
            if (!uploadedFile) return;

            uploadedFile.status = UploadedFile.StatusTypes.ERROR;
        });

        document.body.addEventListener("doc-selection-change", (evt) => {
            const detail = /** @type{CustomEvent} */ (evt).detail;
            let uploadedDocument = this.uploadedFiles?.getFileById(detail.id);

            if (uploadedDocument && !detail.checked) {
                this.removeSelectedFileId(detail.id);
                this.uploadedFiles.removeFile(uploadedDocument)
                return;
            }

            if (!uploadedDocument && detail.checked) {
                this.addSelectedFileId(detail.id);
                uploadedDocument = this.#createFile(detail.name);
                uploadedDocument.fileId = detail.id;
                uploadedDocument.status = UploadedFile.StatusTypes.COMPLETE;

                uploadedDocument.removeElement.addEventListener("click", () => {
                    updateYourDocuments().finally(() => {
                        this.#uncheckDocument(uploadedDocument?.fileId);
                        this.#checkDocuments();
                    });
                });
                return;
            }

            if (!uploadedDocument && !detail.checked) {
                this.removeSelectedFileId(detail.id);
            }
        });

        document.body.addEventListener("file-uploads-processed", this.messageInput?.enableSubmit);
        document.body.addEventListener("file-uploads-removed", () => {
            if (!this.messageInput?.getValue()) this.messageInput.reset();
        });
    }

    /**
     * Create a uploadedFile element
     * @param {String} name - File(s) to be uploaded
     */
    #createFile(name) {
        if (!this.uploadedFiles) {
            const uploadedFiles = /** @type {UploadedFiles} */ (
                document.createElement("uploaded-files")
            );
            this.textarea.prepend(uploadedFiles);

            if (!this.messageInput.getValue()) {
                this.messageInput.reset();
                this.#moveCaretToEnd();
            }
        }

        return this.uploadedFiles.addFile(name);
    }


    /**
     * Process files for upload
     * @param {FileList | null | undefined} files - File(s) to be uploaded
     */
    #handleFiles(files) {
        if (!files) return;

        this.#disableSubmit();

        for (const file of files) {
            this.uploadFile(file);
        }
    }


    /**
     * Triggers file upload to the server with the provided url
     * @param {File} file - File to be uploaded
     */
    uploadFile(file, uploadUrl = this.uploadUrl) {
        if (!file || !uploadUrl || !this.textarea) return;

        const uploadedFile = this.#createFile(file.name);

        uploadedFile.removeElement.addEventListener("click", () => {
            if (uploadedFile.status !== UploadedFile.StatusTypes.COMPLETE) return;
            updateYourDocuments().finally(() => {
                this.#uncheckDocument(uploadedFile.fileId);
                this.#checkDocuments();
            });
        });

        const xhr = new XMLHttpRequest();
        xhr.open("POST", uploadUrl);
        xhr.setRequestHeader("X-CSRFToken", getCsrfToken());

        xhr.upload.addEventListener("progress", (evt) => {
            if (evt.lengthComputable) {
                const percent = Math.round((evt.loaded / evt.total) * 100);
                uploadedFile.progressPercent = percent - 1;
            }
        });

        xhr.onload = () => {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                const responseStatus = response.status.toLowerCase();
                const errorStatus = ![
                    UploadedFile.StatusTypes.COMPLETE,
                    UploadedFile.StatusTypes.PROCESSING,
                ].includes(responseStatus);

                if (response.errors || errorStatus) {
                    uploadedFile.status = UploadedFile.StatusTypes.ERROR;
                    console.error(response.errors);
                }

                if (response.file_id) uploadedFile.fileId = response.file_id;
                if (response.file_name !== file.name) uploadedFile.fileName = response.file_name;

                if (responseStatus === UploadedFile.StatusTypes.COMPLETE) {
                    uploadedFile.status = UploadedFile.StatusTypes.COMPLETE;
                    updateYourDocuments().finally(() => { this.#checkDocuments(response.file_id) });
                }

                if (responseStatus === UploadedFile.StatusTypes.PROCESSING) pollFileStatus(response.file_id);
            } else {
                uploadedFile.status = UploadedFile.StatusTypes.ERROR;
            }
        };

        xhr.onerror = () => uploadedFile.status = UploadedFile.StatusTypes.ERROR;

        const formData = new FormData();
        formData.append("file", file);
        xhr.send(formData);
    }


    /**
     * Check/uncheck the specified document(s)
     * @param {string | null | undefined} id - Document ID
    */
    #checkDocuments(id=null) {
        this.addSelectedFileId(id);
        this.uploadedFileIds.forEach(id => {
            const checkbox = /** @type {HTMLInputElement} */ (document.querySelector(`#file-${id}`));
            if (!checkbox) return;
            checkbox.checked = true;
            checkbox.dispatchEvent(new Event("change", { bubbles: true }));
        })
        this.deselectedFileIds.forEach(id => {
            const checkbox = /** @type {HTMLInputElement} */ (document.querySelector(`#file-${id}`));
            if (!checkbox) return;
            checkbox.checked = false;
            checkbox.dispatchEvent(new Event("change", { bubbles: true }));
        })
    }


    /**
     * Uncheck the specified document
     * @param {string | null | undefined} id - Document ID
    */
    #uncheckDocument(id) {
        this.removeSelectedFileId(id);
        const checkbox = /** @type {HTMLInputElement} */ (document.querySelector(`#file-${id}`));
        if (!checkbox) return;
        checkbox.checked = false;
        checkbox.dispatchEvent(new Event("change", { bubbles: true }));
    }


    /**
     * Moves the caret to the end of the textarea
    */
    #moveCaretToEnd() {
        if (!this.textarea) return;

        this.textarea.focus();
        const range = document.createRange();
        range.selectNodeContents(this.textarea);
        range.collapse(false);
        const selection = window.getSelection();
        selection?.removeAllRanges();
        selection?.addRange(range);
    }
}
customElements.define("file-upload", FileUpload);

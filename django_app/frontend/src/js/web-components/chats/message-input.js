// @ts-check

import { pollFileStatus, updateYourDocuments } from "../../services";
import { hideElement } from "../../utils";

export class MessageInput extends HTMLElement {
  constructor() {
    super();
  }

  connectedCallback() {
    this.#bindEvents();
  }

  #sendMessage() {
    this.#hideWarnings();
    this.uploadedFilesWrapper?.remove();

    if (this.textarea?.textContent?.trim()) {
      this.closest("form")?.requestSubmit();
    }
  }


  #bindEvents(textarea = this.textarea) {
    if (!this.textarea) return;

    this.sendButton?.addEventListener("click", (evt) => {
      evt.preventDefault();
      this.#sendMessage();
    });

    // Submit form on enter-key press (providing shift isn't being pressed)
    textarea.addEventListener("keypress", (evt) => {
      if (evt.key === "Enter" && !evt.shiftKey && this.textarea) {
        evt.preventDefault();
        this.#sendMessage();
      }
    });

    if (textarea.classList.contains("drop-zone")) {
      this.textarea.addEventListener("dragover", (evt) => {
        evt.preventDefault();
        this.textarea.classList.add("drag-over");
      });

      textarea.addEventListener("dragleave", (evt) => {
        this.textarea.classList.remove("drag-over");
      });

      textarea.addEventListener("drop", (evt) => {
        evt.preventDefault();
        this.textarea.classList.remove("drag-over");
        const files = evt.dataTransfer?.files;
        this.#handleFiles(files);
      });
    }

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
        this.querySelector(`[data-id="${id}"]`)
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

      updateYourDocuments().then(() => {this.#checkDocument(id)});
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


  get sendButton() {
    return /** @type {HTMLButtonElement} */ (document.querySelector(".rb-send-button"));
  }


  get textarea() {
    return /** @type {HTMLDivElement} */ (this.querySelector(".message-input"));
  }


  get fileInput() {
    const fileInputSelector = this.getAttribute("file-input-selector");
    if (!fileInputSelector) return null;

    return /** @type {HTMLInputElement} */ (document.querySelector(fileInputSelector));
  }


  get uploadedFilesWrapper() {
    return /** @type {HTMLDivElement} */ (this.querySelector(".uploaded-files-wrapper"));
  }


  get uploadUrl() {
    return this.getAttribute("upload-url");
  }


  uploadFile(file, uploadedFileElement, uploadUrl=this.uploadUrl) {
    if (!uploadUrl) return;

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

        // TODO: Define status types somewhere
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
          updateYourDocuments().then(() => {this.#checkDocument(response.file_id)});
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
  #checkDocument(id, checked=true) {
    if (!id) return;

    const checkbox = /** @type {HTMLInputElement} */ (document.querySelector(`#file-${id}`));
    checkbox.checked = checked;
  }


  #addFileToTextBox(file, textarea=this.textarea) {
    let uploadedFilesWrapper = /** @type {HTMLDivElement} */ (textarea.querySelector(".uploaded-files-wrapper"));
    let uploadedFiles = /** @type {HTMLDivElement} */ (textarea.querySelector(".uploaded-files"));

    if (!uploadedFilesWrapper) {
      const template = /** @type {HTMLTemplateElement} */ (document.getElementById('uploaded-files-template'));
      uploadedFilesWrapper = /** @type {HTMLDivElement} */ (template?.content.firstElementChild?.cloneNode(true));
      uploadedFiles = /** @type {HTMLDivElement} */ (uploadedFilesWrapper.querySelector(".uploaded-files"));
      textarea.prepend(uploadedFilesWrapper);
    }
    let uploadedFileElement = this.#createFileElement(file);
    uploadedFiles.appendChild(uploadedFileElement);
    return uploadedFileElement;
  }


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
      const uploadedFiles = this.textarea.querySelector(".uploaded-files");
      const uploadedFilesWrapper = this.textarea.querySelector(".uploaded-files-wrapper");
      if (uploadedFiles?.innerHTML === "") uploadedFilesWrapper?.remove();
      updateYourDocuments().then(() => {this.#checkDocument(uploadedFileWrapper.dataset.id, false)});
    });
    return uploadedFileWrapper;
  }


  #handleFiles(files) {
    for (const file of files) {
      const uploadedFileElement = this.#addFileToTextBox(file);
      this.uploadFile(file, uploadedFileElement);
    }
    this.textarea.appendChild(document.createTextNode("\n"));
    this.#moveCaretToEnd();
  }


  #moveCaretToEnd(textarea=this.textarea) {
    textarea.focus();
    const range = document.createRange();
    range.selectNodeContents(textarea);
    range.collapse(false);
    const selection = window.getSelection();
    selection?.removeAllRanges();
    selection?.addRange(range);
  }


  /**
   * Returns the current message
   * @returns string
   */
  getValue = () => {
    return this.textarea?.textContent?.trim() || "";
  };


  /**
   * Clears the message
   */
  reset = () => {
    if (this.textarea) this.textarea.textContent = "";
  };


  /**
   * Hides the warning messages displayed under the textarea
   */
  #hideWarnings = () => {
    const chatWarnings = /** @type {HTMLDivElement} */ (
      document.querySelector(".chat-warnings")
    );
    if (chatWarnings) hideElement(chatWarnings);
  };
}
customElements.define("message-input", MessageInput);

// @ts-check

import { emitEvent, Events, listenEvent } from "../../../interaction_design_system/ids/events";
import { refreshUI } from "../../services";

class DocumentSelector extends HTMLElement {
  selectedDocuments = [];

  connectedCallback() {
    if (!this.#getDocuments()) return;

    // update on page load
    this.#getSelectedDocuments();
    this.#bindDocumentListeners();

    // listen for completed docs
    listenEvent(Events.FILE_STATUS_COMPLETE, (evt) => {
      const fileStatus = evt.detail.fileStatus;
      if (!(fileStatus instanceof HTMLElement)) return;

      const completedDoc = fileStatus.closest(
        ".govuk-checkboxes__item"
      );
      if (!completedDoc) return;
      completedDoc.querySelector("file-status")?.remove();
      this.querySelector(".govuk-checkboxes")?.appendChild(completedDoc);
    });

    // listen for deleted docs
    this.#observeDocumentDeletions();
  }

  #getDocuments() {
    this.documents = /** @type {NodeListOf<HTMLInputElement>} */ (
      this.querySelectorAll('input[type="checkbox"]')
    );
    return this.documents;
  }


  #bindDocumentListeners() {
    this.#getDocuments().forEach((doc) => {
      /** @type {HTMLInputElement & { _boundChangeListener?: EventListener}} */
      const el = doc

      if (el._boundChangeListener) {
        // Remove previous listener if present
        el.removeEventListener("change", el._boundChangeListener);
      }

      const listener = (evt) => {
        // update on any selection change
        this.#getSelectedDocuments();
        this.#sendDocSelectionChangeEvent(el);
      };

      el.addEventListener("change", listener);

      // Store reference so we can remove it later
      el._boundChangeListener = listener;
    });
  }


  #getSelectedDocuments() {
    if (!this.documents) return;

      this.selectedDocuments = [];
      this.documents.forEach((document) => {
        if (document.checked) {
          this.selectedDocuments.push({
            id: document.value,
            name: this.querySelector(`[for="${document.id}"]`)?.textContent
          });
        }
      });
      emitEvent(Events.SELECTED_DOCS_CHANGE, this.selectedDocuments);
  }


  #observeDocumentDeletions() {
    this.observer = new MutationObserver((mutations, observer) => {
      for (const mutation of mutations) {
        for (const removedNode of mutation.removedNodes) {
          if (removedNode instanceof HTMLElement) {
            const inputElementChecked = /** @type {HTMLInputElement} */ (
              removedNode.querySelector?.('input[type="checkbox"]:checked')
            );

            if (inputElementChecked) {
              refreshUI(["chat-feed"]);
              this.#sendDocSelectionChangeEvent(inputElementChecked);
              this.#bindDocumentListeners();
              this.#getSelectedDocuments();
            }
          }
        }
      }
    });

    this.observer.observe(this, {
      childList: true,
      subtree: true,
    });
  }


  #sendDocSelectionChangeEvent(inputElement) {
    emitEvent(Events.DOC_SELECTION_CHANGE, {
      id: inputElement.value,
      name: inputElement.title,
      checked: inputElement.checked,
    });
  }

}
customElements.define("document-selector", DocumentSelector);

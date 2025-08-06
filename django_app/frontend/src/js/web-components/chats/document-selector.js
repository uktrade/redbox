// @ts-check

import { updateChatWindow } from "../../services";

class DocumentSelector extends HTMLElement {
  selectedDocuments = [];

  connectedCallback() {
    this.documents = /** @type {NodeListOf<HTMLInputElement>} */ (
      this.querySelectorAll('input[type="checkbox"]')
    );

    if (!this.documents) return;

    // update on page load
    this.#getSelectedDocuments();

    // update on any selection change
    this.documents.forEach((document) => {
      document.addEventListener("change", () => this.#getSelectedDocuments());
    });

    // listen for completed docs
    document.body.addEventListener("doc-complete", (evt) => {
      const completedDoc = /** @type{CustomEvent} */ (evt).detail.closest(
        ".govuk-checkboxes__item"
      );
      completedDoc.querySelector("file-status").remove();
      this.querySelector(".govuk-checkboxes")?.appendChild(completedDoc);
    });

    // listen for deleted docs
    this.#observeDocumentDeletions();
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
      const evt = new CustomEvent("selected-docs-change", {
        detail: this.selectedDocuments,
      });
      document.body.dispatchEvent(evt);
  }


  #observeDocumentDeletions() {
    this.observer = new MutationObserver((mutations, observer) => {
      for (const mutation of mutations) {
        for (const removedNode of mutation.removedNodes) {
          if (
            removedNode instanceof HTMLElement &&
            removedNode.querySelector?.('input[type="checkbox"]:checked')
          ) {
            updateChatWindow();
          }
        }
      }
    });

    this.observer.observe(this, {
      childList: true,
      subtree: true,
    });
  }

}
customElements.define("document-selector", DocumentSelector);

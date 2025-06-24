// @ts-check

import { addShowMore } from "../show-more.js";

class DocumentSelector extends HTMLElement {
  visibleDocumentLimit = 5;
  connectedCallback() {
    const documents = /** @type {NodeListOf<HTMLInputElement>} */ (
      this.querySelectorAll('input[type="checkbox"]')
    );
    addShowMore({
      container: this.querySelector(".your-documents"),
      itemSelector: '.govuk-checkboxes__item',
      itemDisplay: 'flex',
      visibleCount: 5,
    })

    const getSelectedDocuments = () => {
      let selectedDocuments = [];
      documents.forEach((document) => {
        if (document.checked) {
          selectedDocuments.push({
            id: document.value,
            name: this.querySelector(`[for="${document.id}"]`)?.textContent
          });
        }
      });
      const evt = new CustomEvent("selected-docs-change", {
        detail: selectedDocuments,
      });
      document.body.dispatchEvent(evt);
    };

    // update on page load
    getSelectedDocuments();

    // update on any selection change
    documents.forEach((document) => {
      document.addEventListener("change", getSelectedDocuments);
    });

    // listen for completed docs
    document.body.addEventListener("doc-complete", (evt) => {
      const completedDoc = /** @type{CustomEvent} */ (evt).detail.closest(
        ".govuk-checkboxes__item"
      );
      completedDoc.querySelector("file-status").remove();
      this.querySelector(".govuk-checkboxes")?.appendChild(completedDoc);
    });
  }
}
customElements.define("document-selector", DocumentSelector);

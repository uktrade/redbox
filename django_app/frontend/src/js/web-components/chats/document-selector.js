// @ts-check

import { addShowMore } from "../show-more.js";

class DocumentSelector extends HTMLElement {
  visibleDocumentLimit = 5;
  connectedCallback() {
    const documents = /** @type {NodeListOf<HTMLInputElement>} */ (
      this.querySelectorAll('input[type="checkbox"]')
    );
    addShowMore({
      container: this.querySelector(".chat-group-container"),
      itemSelector: '.govuk-checkboxes__item',
      itemDisplay: 'flex',
      visibleCount: 5,
    })
    // this.#limitVisibleDocuments();
    // this.#addShowMoreButton();

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

  
  #limitVisibleDocuments() {
    const documentGroups = this.querySelectorAll(".chat-group-container");
    documentGroups.forEach((group) => {
      const documents = this.querySelectorAll('.govuk-checkboxes__item');
      documents.forEach((document, index) => {
        document.style.display = index < this.visibleDocumentLimit ? "flex" : "none";
      });
      group.dataset.visibleCount = this.visibleDocumentLimit
    });
  }

  #addShowMoreButton() {
    const documentGroups = this.querySelectorAll(".chat-group-container");
    documentGroups.forEach((group) => {
      const documents = this.querySelectorAll('.govuk-checkboxes__item');
      const visibleCount = parseInt(group.dataset.visibleCount, 10) || this.visibleDocumentLimit;
      if (documents.length > visibleCount) {
        const showMoreDiv = document.createElement("div");
        showMoreDiv.id = "show-more-div";
        const showMoreLink = document.createElement("a");
        showMoreLink.textContent = "Show more...";
        showMoreLink.id = "show-more-button";
        showMoreLink.classList.add("rb-chat-history__link", "govuk-link--inverse");

        showMoreLink.addEventListener("click", () => {
          this.#showMoreDocuments(group);
        });

        showMoreDiv.appendChild(showMoreLink);

        group.appendChild(showMoreDiv);
      }
    });
  }

  #showMoreDocuments(group, item_query_selector) {
    const documents = group.querySelectorAll('.govuk-checkboxes__item');
    // let visibleCount = parseInt(group.dataset.visibleCount, 10) || this.visibleDocumentLimit;
    // const newVisibleCount = Math.min(visibleCount + this.visibleDocumentLimit, documents.length);

    documents.forEach((doc) => {
      doc.style.display = "flex";
      // if (index < newVisibleCount) {
      //   doc.style.display = "flex";
      // }
    });

    // group.dataset.visibleCount = newVisibleCount;

    // Hide the button now all documents are visible
    const button = group.querySelector("#show-more-button");
    if (button) {
      button.style.display = "none";
    }
    // if (newVisibleCount >= documents.length) {
    //   const button = group.querySelector("#show-more-button");
    //   if (button) {
    //     button.style.display = "none";
    //   }
    // }
  }
}
customElements.define("document-selector", DocumentSelector);

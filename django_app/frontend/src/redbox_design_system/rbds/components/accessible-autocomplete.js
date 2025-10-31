// @ts-check

import accessibleAutocomplete from 'accessible-autocomplete'

export class AccessibleAutocomplete extends HTMLElement {

  connectedCallback() {
    if (!this.selectElement) return;

    this.initAutocomplete();
    this.initObserver();
  }

  disconnectedCallback() {
    if (this.observer) this.observer.disconnect();
  }


  initAutocomplete() {
    if (!this.selectElement) return;

    if (this.inputElement) {
      const wrapper = this.inputElement.closest(".autocomplete__wrapper");
      const container = wrapper?.parentElement;
      container?.remove();
    }

    accessibleAutocomplete.enhanceSelectElement({
      defaultValue: '',
      selectElement: this.selectElement,
      onConfirm: this.onConfirm,
      confirmOnBlur: false,
      name: `${this.selectElement.name}-input`,
    })

    this.inputElement.type = "search";
  }


  /**
   * Select Element
  */
  get selectElement() {
    if (!this._selectElement || !document.body.contains(this._selectElement)) {
      this._selectElement = /** @type {HTMLSelectElement} */ (
        this.querySelector('select')
      );
    }
    return this._selectElement;
  }


  /**
   * Input Element
  */
  get inputElement() {
    if (!this._inputElement || !document.body.contains(this._inputElement)) {
      this._inputElement = /** @type {HTMLInputElement} */ (
        this.querySelector('input')
      );
    }
    return this._inputElement;
  }


  /**
   * Method to fire changed event for updated select element
  */
  onConfirm(selectedOption) {
    const requestedOption = [].filter.call(this.selectElement?.options, option => (option.textContent || option.innerText) === selectedOption)[0]
    if (requestedOption) {
      requestedOption.selected = true;
      this.selectElement.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }


  /**
   * Setup observer for reinitialising the autocomplete component
  */
  get observer() {
    if (!this._observer) {
      this._observer = new MutationObserver((mutations, observer) => {
        for (const mutation of mutations) {
          if (mutation.type === 'childList') {
            this.initAutocomplete();
          }
        }
      });
    }
    return this._observer;
  }


  /**
   * Bind/unbind observer to select element
  */
  initObserver() {
    if (!this.selectElement) return;
    if (this.observer) this.observer.disconnect();

    this.observer.observe(this.selectElement, {
      childList: true,
      subtree: false,
    });
  }

}
customElements.define("rbds-accessible-autocomplete", AccessibleAutocomplete);

// @ts-check

/**
 * Hide an element by using the govuk-!-display-none class
 * @param {HTMLElement} element - HTML element
*/
export function hideElement(element) {
    element.classList.add("govuk-!-display-none");
}


/**
 * Show an element by removing the govuk-!-display-none class
 * @param {HTMLElement} element - HTML element
*/
export function showElement(element) {
    element.classList.remove("govuk-!-display-none");
}


/**
 * Add a fallback parameter to getAttribute()
 * @param {HTMLElement} elem - element
 * @param {string} attr - attribute name
 * @param {string} fallback - default value
*/
export function getAttributeOrDefault(elem, attr, fallback) {
    return elem.getAttribute(attr) ?? fallback;
}


/**
 * Fetch a numeric string attribute from an element
 * @param {HTMLElement} elem - Element
 * @param {string} attrName - Name of attribute
 * @param {number} fallback - Numeric fallback value
*/
export function getNumericAttr(elem, attrName, fallback) {
    const raw = elem.getAttribute(attrName ?? "");
    const parsed = parseInt(raw ?? "");
    return isNaN(parsed) ? fallback : parsed;
}


/**
 * Returns the CSRF token
 * @returns {String} CSRF token value
*/
export function getCsrfToken() {
    return /** @type {HTMLInputElement | null} */ (
        document.querySelector('[name="csrfmiddlewaretoken"]')
    )?.value || "";
}

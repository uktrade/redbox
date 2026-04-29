// @ts-check

const VISUALLY_HIDDEN_CLASS = "govuk-!-display-none"

/**
 * Hide an element by using the govuk-!-display-none class
 * @param {Element | undefined | null} element - Element
*/
export function hideElement(element) {
    if (element) element.classList.add(VISUALLY_HIDDEN_CLASS);
}


/**
 * Show an element by removing the govuk-!-display-none class
 * @param {Element | undefined | null} element - Element
*/
export function showElement(element) {
    if (element) element.classList.remove(VISUALLY_HIDDEN_CLASS);
}


/**
 * Checks whether an element is hidden
 * @param {Element | undefined | null} element - Element
*/
export function isHidden(element) {
    if (!element) return true;

    return element.classList.contains(VISUALLY_HIDDEN_CLASS);
}


/**
 * Add a fallback parameter to getAttribute()
 * @param {HTMLElement} elem - element
 * @param {string} attr - attribute name
 * @param {string} fallback - default value
*/
export function getAttributeOrDefault(elem, attr, fallback) {
    if (!elem) return fallback;
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

// @ts-check

import DOMPurify from "dompurify";

const HIDDEN_CLASS = "govuk-!-display-none"
const VISUALLY_HIDDEN_CLASS = "govuk-visually-hidden"

/**
 * Hide an element by using the govuk-!-display-none class
 * @param {Element | undefined | null} element - Element
*/
export function hideElement(element) {
    if (element) element.classList.add(HIDDEN_CLASS);
}


/**
 * Show an element by removing the govuk-!-display-none class
 * @param {Element | undefined | null} element - Element
*/
export function showElement(element) {
    if (element) element.classList.remove(HIDDEN_CLASS);
}


/**
 * Visually hide an element by using the govuk-visually-hidden class
 * Remains accessible to screen readers
 * @param {Element | undefined | null} element - Element
*/
export function visuallyHideElement(element) {
    if (element) element.classList.add(VISUALLY_HIDDEN_CLASS);
}


/**
 * Visually show an element by removing the govuk-visually-hidden class
 * Remains accessible to screen readers
 * @param {Element | undefined | null} element - Element
*/
export function visuallyShowElement(element) {
    if (element) element.classList.remove(VISUALLY_HIDDEN_CLASS);
}


/**
 * Checks whether an element is hidden
 * @param {Element | undefined | null} element - Element
*/
export function isHidden(element) {
    if (!element) return true;

    return element.classList.contains(HIDDEN_CLASS);
}


/**
 * Checks whether an element is visible
 * @param {HTMLElement | undefined | null} element - Element
*/
export function isVisible(element) {
    if (!element) return false;
    return (
        !element.classList.contains(HIDDEN_CLASS) &&
        !element.classList.contains(VISUALLY_HIDDEN_CLASS) &&
        element.offsetParent !== null && // visible in layout
        !element.hasAttribute('hidden') &&
        getComputedStyle(element).visibility !== 'hidden'
    );
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


/**
 * Focuses the first focusable element in a container/element
 * @param {HTMLElement | null | undefined} container - Element/Container
*/
export function focusFirstFocusable(container) {
    if (!container) return;

    const selectors = [
        'a[href]',
        'button:not([disabled])',
        'input:not([disabled])',
        'select:not([disabled])',
        'textarea:not([disabled])',
        '[tabindex]:not([tabindex="-1"])',
        '[role="button"]:not([disabled])'
    ];

    // Check if the container itself matches any selector in the list
    const isFocusable = selectors.some(selector => container.matches(selector));

    if (isFocusable && isVisible(container)) return container.focus();

    // Check the child elements
    const elements = Array.from(
        container.querySelectorAll(selectors.join(','))
    );

    const visible = /** @type {HTMLElement} */ (elements.find(el => {
        const htmlEl = /** @type {HTMLElement} */ (el);
        return isVisible(htmlEl);
    }));

    if (visible) visible.focus();
}


/**
 * Returns the currently focused element
 * @returns {Element | null} focused element
*/
export function getFocusedElement() {
    return document.activeElement;
}


/**
 * Sanitize HTML content
 * @param {string} html HTML content
 * @returns {string} Safe html
*/
export function sanitizeHtml(html) {
    return DOMPurify.sanitize(html);
}

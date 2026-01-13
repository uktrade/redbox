// @ts-check

/**
 * Get active tool id if present
*/
export function getActiveToolId() {
    return /** @type {HTMLElement} */ (
        document.querySelector('[data-tool-id]')
    )?.dataset.toolId;
}


/**
 * Get active tool slug if present
*/
export function getActiveToolSlug() {
    return /** @type {HTMLElement} */ (
        document.querySelector('[data-tool-slug]')
    )?.dataset.toolSlug;
}

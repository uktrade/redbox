// @ts-check

/**
 * Get active skill id if present
*/
export function getActiveSkillId() {
    return /** @type {HTMLElement} */ (
        document.querySelector('[data-skill-id]')
    )?.dataset.skillId;
}


/**
 * Get active skill slug if present
*/
export function getActiveSkillSlug() {
    return /** @type {HTMLElement} */ (
        document.querySelector('[data-skill-slug]')
    )?.dataset.skillSlug;
}

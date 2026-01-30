const scrollBlockers = new Set();
const scrollDisabledClass = "no-scroll";
let blockerCounter = 0;

/**
 * Enable no-scroll for the document.
 * Returns a unique ID that should be used to remove it later.
 */
export function enableNoScroll() {
  const id = `scroll-blocker-${blockerCounter++}`;
  scrollBlockers.add(id);
  document.documentElement.classList.add(scrollDisabledClass);
  return id;
}


/**
 * Remove no-scroll for a given blocker ID.
 */
export function disableNoScroll(id) {
  scrollBlockers.delete(id);
  if (scrollBlockers.size === 0) {
    document.documentElement.classList.remove(scrollDisabledClass);
  }
}


/**
 * Toggle no-scroll for a given boolean state.
 * Returns the blocker ID if enabling scroll.
 * @param {boolean} enable on/off state
 * @param {string | null | undefined} id existing scroll ID
 * @returns {String | null} generated scroll ID
 */
export function toggleNoScroll(enable, id = null) {
    if (enable) {
        return enableNoScroll();
    } else if (id) {
        disableNoScroll(id);
    }
    return null;
}

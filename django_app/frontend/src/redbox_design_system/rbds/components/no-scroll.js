const scrollBlockers = new Map(); // id -> source
const scrollDisabledClass = "no-scroll";
const scrollLockSourcesId = "data-scroll-lock-sources";
let blockerCounter = 0;

/**
 * Enable no-scroll for the document.
 * Returns a unique ID that should be used to remove it later.
 * @param {string | undefined} source scroll source
 */
export function enableNoScroll(source="generic") {
  const id =`scroll-blocker-${blockerCounter++}`;
  scrollBlockers.set(id, source);
  document.documentElement.classList.add(scrollDisabledClass);
  updateScrollBlockers();

  return id;
}


/**
 * Remove no-scroll for a given blocker ID.
 * @param {string} id existing scroll-blocker ID
 */
export function disableNoScroll(id) {
  scrollBlockers.delete(id);
  if (scrollBlockers.size === 0) {
    document.documentElement.classList.remove(scrollDisabledClass);
    document.documentElement.removeAttribute(scrollLockSourcesId);
  } else {
    updateScrollBlockers();
  }
}


/**
 * Toggle no-scroll for a given boolean state.
 * Returns the blocker ID if enabling scroll.
 * @param {boolean} enable on/off state
 * @param {string | null | undefined} id scroll-blocker ID
 * @param {string | undefined} source scroll source
 * @returns {String | null} generated scroll-blocker ID if enabled
 */
export function toggleNoScroll(enable, id = null, source=undefined) {
    if (enable) {
        return enableNoScroll(source);
    } else if (id) {
        disableNoScroll(id);
    }
    return null;
}


/**
 * Update scroll-blocker sources on the document.
 */
function updateScrollBlockers() {
  document.documentElement.setAttribute(
    scrollLockSourcesId,
    [...scrollBlockers.values()].join(" ")
  );
}

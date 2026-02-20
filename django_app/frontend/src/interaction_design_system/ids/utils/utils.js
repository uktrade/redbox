// @ts-check

/**
 * Returns the breakpoint px for a given size size (xs/s/m/l/xl/full)
 * @param {String} name - size
*/
export function getBreakpointPx(name) {
    const val = getComputedStyle(document.documentElement)
        .getPropertyValue(`--screen-width-${name}`)
        .trim();
    return parseInt(val, 10);
}


/**
 * Checks if the breakpoint is below a given size
 * @param {String} name - size
 * @returns {Boolean}
*/
export function isBelowBreakpoint(name) {
    return window.innerWidth < getBreakpointPx(name);
}

// @ts-check

function updateHeaderHeight() {
    const header = /** @type {HTMLElement} */ (document.querySelector(".rbds-header-wrapper"));
    if (!header) return;

    document.documentElement.style.setProperty(
        "--header-height",
        `${header.offsetHeight}px`,
    );
}

// Update on load + resize
window.addEventListener("load", updateHeaderHeight);
window.addEventListener("resize", updateHeaderHeight);

// Update on HTMX swaps
document.body.addEventListener("htmx:afterSwap", updateHeaderHeight);
document.body.addEventListener("htmx:afterSettle", updateHeaderHeight);

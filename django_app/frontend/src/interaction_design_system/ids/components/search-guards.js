const DEFAULT_MIN_LENGTH = 2;


document.body.addEventListener("htmx:configRequest", (event) => {
    const element = event.target;

    // only act on search inputs
    if (!(element instanceof HTMLInputElement)) return;
    if (!element.name) return;

    // only apply to query/search inputs
    const isSearchField =
        element.name === "query" ||
        element.name === "q" ||
        element.dataset.minLengthSearch !== undefined;

    if (!isSearchField) return;

    const value = element.value?.trim() || "";
    const minLength = parseInt(element.dataset.minLengthSearch || DEFAULT_MIN_LENGTH);

    if (value.length < minLength) {
        event.preventDefault();
    }
});

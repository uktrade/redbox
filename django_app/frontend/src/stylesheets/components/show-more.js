export function addShowMore({
    container,
    itemSelector,
    visibleCount = 5,
    buttonText = "Show more...",
}) {
    const items = container.querySelectorAll(itemSelector);
    if (items.length < visibleCount) {
        items.forEach((item) => item.style.removeProperty('display'));
        return;
    }

    items.forEach((item, index) => {
        item.style.display = (index >= visibleCount) ? "none" : "";
    });

    // Create show more button
    let showMoreDiv = container.querySelector("#show-more-div");
    if (!showMoreDiv) {
        showMoreDiv = document.createElement("div");
        showMoreDiv.id = "show-more-div";
        const showMoreLink = document.createElement("a");
        showMoreLink.textContent = buttonText;
        showMoreLink.classList.add(
            "show-more-link",
            "govuk-link",
            "govuk-link--no-visited-state",
            "govuk-link--no-underline"
        );

        // Show all items and remove the button when clicked
        showMoreLink.addEventListener("click", () => {
            items.forEach((item) => item.style.removeProperty('display'));
            showMoreDiv.remove();
        });

        // Create element
        showMoreDiv.appendChild(showMoreLink);
        container.appendChild(showMoreDiv);
    }
}

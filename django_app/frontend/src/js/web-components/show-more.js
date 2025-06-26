export function addShowMore({
    container,
    itemSelector,
    visibleCount = 5,
    buttonText = "Show more...",
}) {
    const items = container.querySelectorAll(itemSelector);
    if (items.length < visibleCount) return;

    // Store display setting
    const displayValue = items[0].style.display;
    let displayValues = [];

    items.forEach((item, index) => {
        // Preserve display value
        displayValues[index] = item.style.display;

        // Hide additional items
        if (index >= visibleCount) item.style.display = "none";
      });

    // Create show more button
    const showMoreDiv = document.createElement("div");
    showMoreDiv.id = "show-more-div";
    const showMoreLink = document.createElement("a");
    showMoreLink.textContent = buttonText;
    showMoreLink.id = "show-more-button";
    showMoreLink.classList.add("rb-chat-history__link", "govuk-link--inverse");

    // Show all items and remove the button when clicked
    showMoreLink.addEventListener("click", () => {
        items.forEach((item, index) => {item.style.display = displayValues[index]});
        showMoreLink.remove();
    });

    // Create element
    showMoreDiv.appendChild(showMoreLink);
    container.appendChild(showMoreDiv);
}

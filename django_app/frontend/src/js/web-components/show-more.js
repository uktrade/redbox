export function addShowMore({
    container,
    itemSelector,
    itemDisplay = "block",
    visibleCount = 5,
    buttonText = "Show more...",
}) {
    const items = container.querySelectorAll(itemSelector);
    if (items.length < visibleCount) return;

    // Hide additional items
    items.forEach((item, index) => {
        item.style.display = index < visibleCount ? itemDisplay : "none";
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
        items.forEach((item) => {item.style.display = itemDisplay});
        showMoreLink.remove();
    });

    // Create element
    showMoreDiv.appendChild(showMoreLink);
    container.appendChild(showMoreDiv);
}
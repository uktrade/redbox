const collapsibleMenu = () => {
    document.addEventListener("DOMContentLoaded", () => {
        const menuButton = document.querySelector(".govuk-header__menu-button");
        const headerList = document.querySelector(".header-list");

        const toggleMenu = () => {
            const expanded = menuButton.getAttribute("aria-expanded") === "true";
            menuButton.setAttribute("aria-expanded", (!expanded).toString());
            headerList.style.display = (!expanded) ? "block" : "none"
        }

        menuButton.addEventListener("click", toggleMenu)
    })
}

collapsibleMenu()

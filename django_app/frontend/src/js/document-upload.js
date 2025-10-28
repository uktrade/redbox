import "./web-components/documents/upload-button.js";
import { hideElement, showElement } from "./utils";

const individualTeamVisibility = () => {
    document.addEventListener("DOMContentLoaded", () => {
        const visibilityGroup = document.getElementById("visibility-group");
        const teamGroup = document.getElementById("team-group");
        const individualRadio = document.getElementById("visibility-individual");
        const teamRadio = document.getElementById("visibility-team");

        if (!visibilityGroup || !teamGroup || !individualRadio || !teamRadio) return;

        const toggleTeamGroup = () => {
            teamRadio.checked ? showElement(teamGroup) : hideElement(teamGroup);
        };

        individualRadio.addEventListener("change", toggleTeamGroup);
        teamRadio.addEventListener("change", toggleTeamGroup);
    });
};

individualTeamVisibility();

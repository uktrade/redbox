import { deleteAllDrafts } from "./draft-prompt-storage"

document.addEventListener("click", (event) => {
  if (
    event.target instanceof Element &&
    event.target.closest("[data-clear-draft-prompts]")
  ) {void deleteAllDrafts()}
})

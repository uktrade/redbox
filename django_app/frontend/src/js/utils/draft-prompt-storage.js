const DRAFT_PREFIX = "redbox:draft-prompt:"
const MAX_DRAFT_AGE = 3 * 60 * 1000 // 3 minutes

function getDraftKey ({ userId, chatKey }) {
  return `${DRAFT_PREFIX} ${encodeURIComponent(userId)} ${encodeURIComponent(chatKey)}`;
}

export async function saveDraft ({ userId, chatKey, text }) {
  if (!userId || !chatKey) return false

  try {
    const key = getDraftKey({ userId, chatKey });

    if (!text.trim()) {
      sessionStorage.removeItem(key);
      return true;
    }

    sessionStorage.setItem(
      key,
      JSON.stringify({
        text,
        savedAt: Date.now()
      })
    );

    return true;

  } catch (error) {
    console.warn("Could not save draft prompt", error);
    return false;
  }
};


export async function loadDraft ({ userId, chatKey }) {
  if (!userId || !chatKey) return null

  try {
    const key = getDraftKey({ userId, chatKey });
    const storedDraftJson = sessionStorage.getItem(key);

    if (!storedDraftJson) return null

    const draft = JSON.parse(storedDraftJson);

    const valid =
      typeof draft?.text === "string" &&
      typeof draft?.savedAt === "number" &&
      draft.savedAt <= Date.now() &&
      Date.now() - draft.savedAt < MAX_DRAFT_AGE;

    if (!valid) {
      sessionStorage.removeItem(key);
      return null;
    }

    return draft.text;

  } catch (error) {
    console.warn("Could not load draft prompt", error);
    return null;
  }
};

export async function deleteDraft ({ userId, chatKey }) {
  if (!userId || !chatKey) return false

  try {
    sessionStorage.removeItem(
      getDraftKey({ userId, chatKey })
    );
    return true;
  } catch (error) {
    console.warn("Could not delete draft prompt", error);
    return false;
  }
}

export async function deleteAllDrafts ({ userId, chatKey }) {
  try {
    Object.keys(sessionStorage)
      .filter((key) => key.startsWith(DRAFT_PREFIX))
      .forEach((key) => sessionStorage.removeItem(key));

    return true;

  } catch (error) {
    console.warn("Could not delete all draft prompts", error);
    return false;
  }
}

// @ts-check

/**
 * Checks the status of a uploaded document at regular intervals
 * @param {string} id - Document ID
 * @param {number} retries - Number of retries
*/
 export async function pollFileStatus(id, retries=0) {
    console.log("pollFileStatus ID: ", id);
    const FILE_STATUS_ENDPOINT = "/file-status";
    const CHECK_INTERVAL_MS = 6000;
    const MAX_RETRIES = 100;

    const response = await fetch(`${FILE_STATUS_ENDPOINT}?id=${id}`);
    const responseObj = await response.json();
    const responseStatus = responseObj.status.toLowerCase();

    switch(responseStatus) {
        case "complete":
            document.body.dispatchEvent(new CustomEvent("doc-complete", {
                detail: {id, status: responseStatus},
            }));
        case "processing":
            if (retries >= MAX_RETRIES) break;
            window.setTimeout(() => {
                pollFileStatus(id, retries+1)
            }, CHECK_INTERVAL_MS);
        default:
            document.body.dispatchEvent(new CustomEvent("doc-error", {
                detail: {id, status: responseStatus},
            }));
    }
}

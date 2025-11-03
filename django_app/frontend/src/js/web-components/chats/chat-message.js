// @ts-check

import "../loading-message.js";

window.addEventListener('load', () => {
  const scrollPosition = sessionStorage.getItem('scrollPosition');
  if (scrollPosition !== null) {
    window.scrollTo({
      top: parseInt(scrollPosition),
      behavior: 'instant'
    });
    sessionStorage.removeItem('scrollPosition');
  }
});

export class ChatMessage extends HTMLElement {

  connectedCallback() {
    this.programmaticScroll = false;
    this.streamedContent = "";
    this.#loadMessage();
  }

  #loadMessage = () => {
    const uuid = crypto.randomUUID();
    this.innerHTML = `
            <div class="redbox-message-container govuk-inset-text ${this.dataset.role == 'user' ? `govuk-inset-text-right`: ''} govuk-body" data-role="${
              this.dataset.role
            }" tabindex="-1" id="chat-message-${this.dataset.id}">
                <markdown-converter class="rbds-chat-message__text">${
                  this.dataset.text || ""
                }</markdown-converter>
                ${
                  !this.dataset.text
                    ? `
                      <loading-message data-aria-label="Loading message"></loading-message>
                      <div class="rb-loading-complete govuk-visually-hidden" aria-live="assertive"></div>
                    `
                    : ""
                }
                <sources-list data-id="${uuid}"></sources-list>
                <div class="govuk-error-summary" data-module="govuk-error-summary" hidden>
                  <div class="govuk-body govuk-!-font-weight-bold">
                    There was an unexpected error communicating with Redbox. Please try again.
                  </div>
                  <div class="govuk-body">
                      If the problem persists, please contact
                        <a class="govuk-body govuk-link" href="/support/">support</a>
                    </div>
                </div>
            ${this.dataset.role == 'ai' ?
              `<div class="chat-actions-container">
            </div>`
            : ''}


        `;

    // ensure new chat-messages aren't hidden behind the chat-input
    this.programmaticScroll = true;
    this.scrollIntoView({ block: "end" });

    // Insert route_display HTML
    if (this.dataset.role == "ai") {
      const routeTemplate = /** @type {HTMLTemplateElement} */ (
        document.querySelector("#template-route-display")
      );
      const routeClone = document.importNode(routeTemplate.content, true);
      const actionsContainer = this.querySelector(".chat-actions-container");
      if (actionsContainer) {
        this.appendChild(routeClone);
      }
    }
  };



  #addFootnotes = (content, chatId) => {
    let footnotes = this.querySelectorAll("sources-list a[data-text]");
    footnotes.forEach((footnote, footnoteIndex) => {
      const matchingText = footnote.getAttribute("data-text");
      if (!matchingText || !this.responseContainer) {
        return;
      }
      /*
      this.responseContainer?.update(
        content.replace(matchingText, `${matchingText}<a href="#${footnote.id}" aria-label="Footnote ${footnoteIndex + 1}">[${footnoteIndex + 1}]</a>`)
      );
      */
      this.responseContainer.innerHTML =
        this.responseContainer.innerHTML.replace(
          matchingText,
          `${matchingText}<a class="rb-footnote-link" href="/citations/${chatId}/#${
            footnote.id
          }" aria-label="Footnote ${footnoteIndex + 1}">${
            footnoteIndex + 1
          }</a>`
        );
    });
  };

  /**
   * Displays a chosen file below a message
   * @param {string} fileName
   */
  addFile = (fileName) => {
    let attachedFiles = document.createElement("p");

    attachedFiles.classList.add('govuk-body', 'govuk-!-text-align-right');
    attachedFiles.innerHTML= `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <g clip-path="url(#clip0_771_609)">
    <path d="M19.5 22.5H8.25C7.85232 22.4995 7.47105 22.3414 7.18984 22.0602C6.90864 21.7789 6.75046 21.3977 6.75 21V16.5H8.25V21H19.5V4.5H12.75V3H19.5C19.8977 3.00046 20.279 3.15864 20.5602 3.43984C20.8414 3.72105 20.9995 4.10232 21 4.5V21C20.9995 21.3977 20.8414 21.7789 20.5602 22.0602C20.279 22.3414 19.8977 22.4995 19.5 22.5Z" fill="black"/>
    <path d="M18 7.5H12.75V9H18V7.5Z" fill="black"/>
    <path d="M18 11.25H12V12.75H18V11.25Z" fill="black"/>
    <path d="M18 15H11.25V16.5H18V15Z" fill="black"/>
    <path d="M6.75 14.25C5.75577 14.2489 4.80258 13.8535 4.09956 13.1504C3.39653 12.4474 3.00109 11.4942 3 10.5V2.25H4.5V10.5C4.5 11.0967 4.73705 11.669 5.15901 12.091C5.58097 12.5129 6.15326 12.75 6.75 12.75C7.34674 12.75 7.91903 12.5129 8.34099 12.091C8.76295 11.669 9 11.0967 9 10.5V3.75C9 3.55109 8.92098 3.36032 8.78033 3.21967C8.63968 3.07902 8.44891 3 8.25 3C8.05109 3 7.86032 3.07902 7.71967 3.21967C7.57902 3.36032 7.5 3.55109 7.5 3.75V11.25H6V3.75C6 3.15326 6.23705 2.58097 6.65901 2.15901C7.08097 1.73705 7.65326 1.5 8.25 1.5C8.84674 1.5 9.41903 1.73705 9.84099 2.15901C10.2629 2.58097 10.5 3.15326 10.5 3.75V10.5C10.4989 11.4942 10.1035 12.4474 9.40044 13.1504C8.69742 13.8535 7.74423 14.2489 6.75 14.25Z" fill="black"/>
    </g>
    <defs>
    <clipPath id="clip0_771_609">
    <rect width="24" height="24" fill="white"/>
    </clipPath>
    </defs>
    </svg>  ${fileName}`
    this.insertAdjacentElement("afterend", attachedFiles)

  };

  /**
   * Streams an LLM response
   * @param {string} message
   * @param {string[]} selectedDocuments An array of IDs
   * @param {string[]} activities
   * @param {string} llm
   * @param {string | undefined} sessionId
   * @param {string} endPoint
   * @param {HTMLElement} chatControllerRef
   */
  stream = (
    message,
    selectedDocuments,
    activities,
    llm,
    sessionId,
    endPoint,
    chatControllerRef
  ) => {
    // Scroll behaviour - depending on whether user has overridden this or not
    let scrollOverride = false;
    const is_new_chat = !sessionId;

    function reloadAtCurrentPosition() {
      sessionStorage.setItem('scrollPosition', window.scrollY.toString());
      location.reload();
    }

    window.addEventListener("scroll", (evt) => {
      if (this.programmaticScroll) {
        this.programmaticScroll = false;
        return;
      }
      scrollOverride = true;
    });

    window.addEventListener('load', () => {
      const scrollPosition = sessionStorage.getItem('scrollPosition');
      if (scrollPosition !== null) {
        window.scrollTo(0, parseInt(scrollPosition));
        sessionStorage.removeItem('scrollPosition');
      }
    });

    this.responseContainer =
      /** @type {import("../markdown-converter").MarkdownConverter} */ (
        this.querySelector("markdown-converter")
      );
    let sourcesContainer = /** @type {import("./sources-list").SourcesList} */ (
      this.querySelector("sources-list")
    );
    /** @type {import("./feedback-buttons").FeedbackButtons | null} */
    let responseLoading = /** @type HTMLElement */ (
      this.querySelector(".rb-loading-ellipsis")
    );
    let actionsContainer = this.querySelector(".chat-actions-container")
    let responseComplete = this.querySelector(".rb-loading-complete");
    let webSocket = new WebSocket(endPoint);

    // Stop streaming on escape-key or stop-button press
    const stopStreaming = () => {
      this.dataset.status = "stopped";
      webSocket.close();
    };
    this.addEventListener("keydown", (evt) => {
      if (evt.key === "Escape" && this.dataset.status === "streaming") {
        stopStreaming();
      }
    });
    document.addEventListener("stop-streaming", stopStreaming);

    webSocket.onopen = (event) => {
      webSocket.send(
        JSON.stringify({
          message: message,
          sessionId: sessionId,
          selectedFiles: selectedDocuments,
          activities: activities,
          llm: llm,
        })
      );
      this.dataset.status = "streaming";
      const chatResponseStartEvent = new CustomEvent("chat-response-start");
      document.dispatchEvent(chatResponseStartEvent);
    };

    webSocket.onerror = (event) => {
      if (!this.responseContainer) {
        return;
      }
      this.responseContainer.innerHTML =
        "There was a problem. Please try sending this message again.";
      this.dataset.status = "error";
    };

    webSocket.onclose = (event) => {
      responseLoading.style.display = "none";
      if (responseComplete) {
        responseComplete.textContent = "Response complete";
      }
      if (this.dataset.status !== "stopped") {
        this.dataset.status = "complete";
      }
      const stopStreamingEvent = new CustomEvent("stop-streaming");
      document.dispatchEvent(stopStreamingEvent);
    };

    webSocket.onmessage = (event) => {
      let response;
      try {
        response = JSON.parse(event.data);
      } catch (err) {
        console.log("Error getting JSON response", err);
      }

      if (response.type === "text") {
        this.streamedContent += response.data;
        this.responseContainer?.update(this.streamedContent);
      } else if (response.type === "session-id") {
        chatControllerRef.dataset.sessionId = response.data;
      } else if (response.type === "source") {
        sourcesContainer.add(
          response.data.file_name,
          response.data.url,
          response.data.text_in_answer || ""
        );
      } else if (response.type === "route") {
        // Update the route text on the page now the selected route is known
        let route = this?.querySelector(".redbox-message-route");
        let routeText = route?.querySelector(".route-text");
        if (route && routeText) {
          routeText.textContent = response.data;
          route.removeAttribute("hidden");
        }
      } else if (response.type === "end") {
        // Assign the new message its ID straight away
        const chatMessage = this.querySelector('.govuk-inset-text');
        if (chatMessage) {chatMessage.id = `chat-message-${response.data.message_id}`}
        // Add in feedback and copy buttons dynamically
        if (actionsContainer) {
          const feedbackButtons = document.createElement('feedback-buttons')
          feedbackButtons.dataset.id = response.data.message_id

          const copyText = document.createElement('copy-text')
          copyText.dataset.id = response.data.message_id

          actionsContainer.appendChild(feedbackButtons)
          actionsContainer.appendChild(copyText)

      }
        // this.#addFootnotes(this.streamedContent, response.data.message_id);
        const chatResponseEndEvent = new CustomEvent("chat-response-end", {
          detail: {
            title: response.data.title,
            session_id: response.data.session_id,
            is_new_chat,
          },
        });
        document.dispatchEvent(chatResponseEndEvent);
      } else if (response.type === "error") {
        this.querySelector(".govuk-error-summary")?.removeAttribute(
          "hidden"
        );
        let errorContentContainer = this.querySelector(
          ".govuk-error-summary__title"
        );
        if (errorContentContainer) {
          errorContentContainer.innerHTML = response.data;
        }
      }

      // ensure new content isn't hidden behind the chat-input
      // but stop scrolling if message is at the top of the screen
      if (!scrollOverride) {
        const TOP_POSITION = 88;
        const boxInfo = this.getBoundingClientRect();
        const newTopPosition =
          boxInfo.top -
          (boxInfo.height - (this.previousHeight || boxInfo.height));
        this.previousHeight = boxInfo.height;
        if (newTopPosition > TOP_POSITION) {
          this.programmaticScroll = true;
          this.scrollIntoView({ block: "end" });
        } else {
          scrollOverride = true;
          this.scrollIntoView();
          window.scrollBy(0, -TOP_POSITION);
        }
      }
    };
  };
}
customElements.define("chat-message", ChatMessage);

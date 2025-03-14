// @ts-check
class SendMessage extends HTMLElement {
  showConvertingSpinner() {
    const textArea = document.querySelector("#message");
    if (textArea) {
      textArea.innerText = "";
      textArea.style.position = "relative";
      textArea.style.display = "flex";
      
      const spinnerContainer = document.createElement("div");
      spinnerContainer.className = "spinner-container"
      spinnerContainer.style.cssText = `
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        width: 100%;
        position: absolute;
        top: 0;
        left: 0;
      `;
      
      const spinner = document.createElement("div");
      spinner.className = "spinner";
      spinner.style.cssText = `
        width: 24px;
        height: 24px;
        border: 3px solid #ccc;
        border-top: 3px solid #000;
        border-radius: 50%;
        animation-name: spin;
        animation-duration: 4s;
        animation-iteration-count: infinite;
        margin-right: 8px;
      `;
      
      const text = document.createElement("span");
      text.innerText = "Converting to text";
      spinnerContainer.appendChild(spinner);
      spinnerContainer.appendChild(text);
      textArea.appendChild(spinnerContainer);
    }
  }

  clearConvertingSpinner() {
    const textArea = document.querySelector("#message");
    if (textArea) {
      textArea.innerHTML = "";
      textArea.style.position = "";
    }
  }

  connectedCallback() {
    const sendButtonHtml = `
      <button class="iai-chat-input__button iai-icon-button rb-send-button" type="submit">
        <svg width="28" height="28" viewBox="32 16 29 29" fill="none" focusable="false" aria-hidden="true"><g filter="url(#A)"><use xlink:href="#C" fill="#edeef2"/></g><g filter="url(#B)"><use xlink:href="#C" fill="#fff"/></g><path d="M47.331 36.205v-8.438l3.89 3.89.972-1.007-5.556-5.556-5.556 5.556.972.972 3.889-3.854v8.438h1.389z" fill="currentColor"/><defs><filter x="17" y="1" width="65" height="65" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB"><feFlood flood-opacity="0" result="A"/><feColorMatrix in="SourceAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0"/><feOffset dx="3" dy="3"/><feGaussianBlur stdDeviation="10"/><feColorMatrix values="0 0 0 0 0.141176 0 0 0 0 0.254902 0 0 0 0 0.364706 0 0 0 0.302 0"/><feBlend in2="A"/><feBlend in="SourceGraphic"/></filter><filter id="B" x="0" y="-16" width="85" height="85" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB"><feFlood flood-opacity="0" result="A"/><feColorMatrix in="SourceAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0"/><feOffset dx="-4" dy="-4"/><feGaussianBlur stdDeviation="15"/><feColorMatrix values="0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0"/><feBlend in2="A"/><feBlend in="SourceGraphic"/></filter><path id="C" d="M59 30.5C59 23.596 53.404 18 46.5 18S34 23.596 34 30.5 39.596 43 46.5 43 59 37.404 59 30.5z"/></defs></svg>
        Send
      </button>
    `;
    const stopButtonHtml = `
      <button class="iai-chat-input__button iai-icon-button rb-send-button" type="button">
        <div class="rb-square-icon"></div>
        Stop
      </button>
    `;
    this.innerHTML += sendButtonHtml;
    this.innerHTML += stopButtonHtml;

    this.buttonRecord = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(1)")
    );
    this.buttonSend = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(2)")
    );
    this.buttonStop = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(3)")
    );

    this.buttonSend.style.display = "none";
    this.buttonStop.style.display = "none";

    this.mediaRecorder = null;
    this.audioChunks = [];
    this.isRecording = false;
    this.isStreaming = false;

    this.buttonRecord.addEventListener("click", this.startRecording.bind(this));
    this.buttonStop.addEventListener("click", this.stopAction.bind(this));
    this.buttonSend.addEventListener("click", (e) => {
      e.preventDefault();
      const form = this.closest("form");
      if (form) {
        form.requestSubmit();
      }
    });

    const textArea = document.querySelector("#message");
    if (textArea) {
      textArea.addEventListener("input", () => {
        if (textArea.innerText.trim()) {
          this.showSendButton();
        } else {
          this.showRecordButton();
        }
      });
    }

    document.addEventListener("chat-response-start", () => {
      this.isStreaming = true;
      this.buttonSend.style.display = "none";
      this.buttonRecord.style.display = "none";
      this.buttonStop.style.display = "flex";
    });

    document.addEventListener("chat-response-end", () => {
      this.isStreaming = false;
      this.buttonStop.style.display = "none";
      const textArea = document.querySelector("#message");
      if (textArea && textArea.innerText.trim()) {
        this.showSendButton();
      } else {
        this.showRecordButton();
      }
    });
    document.addEventListener("stop-streaming", () => {
      this.isStreaming = false;
      this.showRecordButton();
    });
  }

  async startRecording() {
    if (this.isRecording || this.isStreaming) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert("Microphone usage is not permitted");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaRecorder = new MediaRecorder(stream);

      this.audioChunks = [];
      this.mediaRecorder.ondataavailable = (event) => {
        this.audioChunks.push(event.data);
      };

      this.mediaRecorder.onstop = async () => {
        this.buttonRecord.style.display = "none";
        this.buttonStop.style.display = "none";
        this.buttonSend.style.display = "none";
        const audioBlob = new Blob(this.audioChunks, { type: "audio/webm" });
        const formData = new FormData();
        formData.append("audio", audioBlob, "recorded_audio.webm");

        try {
          const response = await fetch("/api/transcribe/", {
            method: "POST",
            body: formData,
          });
          const data = await response.json();
          
          this.clearConvertingSpinner();
          
          const textArea = document.querySelector("#message");
          if (textArea) {
            textArea.innerText = data.transcription;
            if (data.transcription) {
              this.showSendButton();
            } else {
              alert(data.error.toString());
              this.showRecordButton();
            }
          }
        } catch (error) {
          this.clearConvertingSpinner();
          console.error("Error:", error);
          this.showRecordButton();
        }

        stream.getTracks().forEach((track) => track.stop());
      };

      this.mediaRecorder.start();
      console.log("Starting recording");
      this.buttonRecord.style.display = "none";
      this.buttonStop.style.display = "flex";
      this.buttonSend.style.display = "none";
      this.isRecording = true;
    } catch (error) {
      alert("Microphone usage is not permitted");
    }
  }

  stopAction() {
    if (this.isRecording && this.mediaRecorder) {
      console.log("Stopped recording");
      this.mediaRecorder.stop();
      this.isRecording = false;
      this.showConvertingSpinner();
    } else if (this.isStreaming) {
      console.log("Stopping stream");
      document.dispatchEvent(new CustomEvent("stop-streaming"));
    }

    this.buttonStop.style.display = "none";
    this.buttonRecord.style.display = "none";
    this.buttonSend.style.display = "none";
  }

  showSendButton() {
    if (this.isRecording || this.isStreaming) return;
    if (!this.buttonRecord || !this.buttonSend || !this.buttonStop) return;

    this.buttonRecord.style.display = "none";
    this.buttonSend.style.display = "flex";
    this.buttonStop.style.display = "none";
  }

  showRecordButton() {
    if (!this.buttonRecord || !this.buttonSend || !this.buttonStop) return;

    this.buttonRecord.style.display = "flex";
    this.buttonSend.style.display = "none";
    this.buttonStop.style.display = "none";
  }
}

customElements.define("send-message", SendMessage);
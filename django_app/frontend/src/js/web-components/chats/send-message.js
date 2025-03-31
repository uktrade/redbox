// @ts-check

class SendMessage extends HTMLElement {
  showConvertingSpinner() {
    const textArea = document.querySelector("#message");
    if (textArea) {
      textArea.className = 'govuk-textareax iai-chat-input__input js-user-text spinning';
      textArea.innerHTML = '      Converting to text...'
    }
  }

  clearConvertingSpinner() {
    const textArea = document.querySelector("#message");
    if (textArea) {
      textArea.innerHTML = "";
      textArea.className = 'govuk-textareax iai-chat-input__input js-user-text';
    }
  }

  connectedCallback() {

    this.buttonRecord = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(1)")
    );
    this.buttonSend = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(4)")
    );
    this.buttonStop = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(2)")
    );
    this.buttonSendStop = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(5)")
    );

    this.buttonStop.style.display = "none";
    this.buttonSendStop.style.display = "none";

    this.mediaRecorder = null;
    this.audioChunks = [];
    this.isRecording = false;
    this.isStreaming = false;

    this.buttonRecord.addEventListener("click", this.startRecording.bind(this));
    this.buttonStop.addEventListener("click", this.stopAction.bind(this));
    this.buttonSendStop.addEventListener("click", this.stopAction.bind(this));
    this.buttonSend.addEventListener("click", (e) => {
      const form = this.closest("form");
      try {
        e.preventDefault();
        form.requestSubmit();
      } catch (error) {
          console.error("Error:", error);
        }
    });

    const textArea = document.querySelector("#message");
    if (textArea) {
      textArea.addEventListener("input", () => {
        if (!textArea.innerHTML) {
          this.showRecordButton();
          this.buttonRecord.disabled = false
        }
      });
    }
    

    document.addEventListener("chat-response-start", () => {
      this.isStreaming = true;
      this.buttonSend.style.display = "none";
      this.buttonSendStop.style.display = "inline";
    });

    document.addEventListener("chat-response-end", () => {
      this.isStreaming = false;
      this.buttonSendStop.style.display = "none";
      const textArea = document.querySelector("#message");
      if (!textArea.innerHTML) {
        this.showRecordButton();
        this.buttonRecord.disabled = false
      }
    });
    document.addEventListener("stop-streaming", () => {
      this.isStreaming = false;
      this.showRecordButton();
      this.buttonRecord.disabled = false
      this.buttonSendStop.style.display = "none";
      this.buttonSend.style.display = "inline";
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
        this.buttonRecord.style.display = "inline";
        this.buttonStop.style.display = "none";
        const audioBlob = new Blob(this.audioChunks, { type: "audio/webm" });
        const formData = new FormData();
        formData.append("audio", audioBlob, "recorded_audio.webm");

        try {
          const response = await fetch("/api/transcribe/", {
            method: "POST",
            body: formData,
          });
          const data = await response.json();
          
          this.showRecordButton();
          this.buttonRecord.disabled = true
          this.clearConvertingSpinner();
          
          const textArea = document.querySelector("#message");
          if (textArea) {
            textArea.innerHTML = data.transcription;
            if (!data.transcription) {
              alert(data.error.toString());
            }
            this.showRecordButton();
            this.buttonRecord.disabled = false
          }
        } catch (error) {
          this.clearConvertingSpinner();
          console.error("Error:", error);
          this.showRecordButton();
          this.buttonRecord.disabled = false
        }

        stream.getTracks().forEach((track) => track.stop());
      };

      this.mediaRecorder.start();
      console.log("Starting recording");
      const textArea = document.querySelector('#message')
      textArea.className = 'govuk-textareax js-user-text iai-chat-input__input recording';
      textArea.innerHTML = '      Listening...'
      this.buttonRecord.style.display = "none";
      this.buttonStop.style.display = "inline";
      this.isRecording = true;
    } catch (error) {
      alert("Microphone usage is not permitted");
    }
  }

  stopAction() {
    this.buttonRecord.style.display = "inline";
    this.buttonSend.style.display = "inline";
    this.buttonStop.style.display = "none";
    this.buttonSendStop.style.display = "none";

    if (this.isRecording && this.mediaRecorder) {
      console.log("Stopped recording");
      this.mediaRecorder.stop();
      this.isRecording = false;
      this.showRecordButton();
      this.buttonRecord.disabled = true
      this.showConvertingSpinner();
    } else if (this.isStreaming) {
      this.showRecordButton();
      this.buttonRecord.disabled = true
      console.log("Stopping stream");
      document.dispatchEvent(new CustomEvent("stop-streaming"));
    }
  }

  showRecordButton() {
    if (!this.buttonRecord || !this.buttonStop) return;

    this.buttonRecord.style.display = "inline";
    this.buttonStop.style.display = "none";
  }
  customElements.define("send-message", SendMessage);
// @ts-check
import { hideElement, showElement } from "../../utils";
import { MessageInput } from "./message-input";

export class SendMessageWithDictation extends HTMLElement {
  connectedCallback() {
    this._apiKey = this.getAttribute("data-api-key");
    this.removeAttribute("data-api-key");

    this.buttonRecord = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(1)")
    );
    this.buttonRecordStop = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(2)")
    );
    this.buttonSend = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(3)")
    );
    this.buttonSendStop = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(4)")
    );

    if (!this.buttonRecord || !this.buttonRecordStop || !this.buttonSend || !this.buttonSendStop || !this.messageInput) return;

    customElements.whenDefined("ids-message-input").then(() => this.#bindEvents())
  }

  #bindEvents() {
    this.showRecordButton();
    this.showSendButton();

    this.mediaRecorder = null;
    this.audioChunks = [];
    this.isStreaming = false;

    this.buttonRecord?.addEventListener("click", this.startRecording.bind(this));
    this.buttonRecordStop?.addEventListener("click", this.stopResponse.bind(this));
    this.buttonSendStop?.addEventListener("click", this.stopResponse.bind(this));
    this.buttonSend?.addEventListener("click", (e) => {
      const form = this.closest("form");
      try {
        e.preventDefault();
        form?.requestSubmit();
      } catch (error) {
        console.error("Error:", error);
      }
    });

    this.messageInput.textarea.addEventListener("input", () => {
      if (!this.messageInput?.getValue() && !this.isStreaming) {
        this.showRecordButton();
        if (this.buttonRecord) this.buttonRecord.disabled = false;
      }
    });

    document.addEventListener("chat-response-start", () => {
      this.isStreaming = true;
      if (this.isRecording) this.stopRecording();
      this.disableSubmit();
      this.hideSendButton();
    });

    document.addEventListener("chat-response-end", () => {
      this.isStreaming = false;
      this.enableSubmit();
      this.showRecordButton();
      this.showSendButton();
    });

    document.addEventListener("stop-streaming", () => {
      this.isStreaming = false;
      this.enableSubmit();
      this.showRecordButton();
      this.showSendButton();
    });
  }

  get messageInput() {
    return /** @type {MessageInput} */ (document.querySelector('ids-message-input'));
  }

  async startRecording() {
    if (this.isStreaming) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert("Microphone usage is not permitted");
      return;
    }

    this.messageInput?.reset();
    this.isRecording = true;
    this.isStreaming = true;
    this.hideRecordButton();
    this.lastFinalTranscript = "";

    const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/transcribe/`;
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log("Transcription WS connected");
    };

    this.ws.onmessage = (event) => {
      console.log("WS Received raw:", event.data);
      try {
        const data = JSON.parse(event.data);
        console.log("Parsed transcript data:", data);
        if (data.type === "transcript") {
          console.log(`Got transcript: ${data.is_partial ? 'PARTIAL' : 'FINAL'} ${data.text}`);
          this._updateTranscript(data.text, data.is_partial);
        } else if (data.type === "error") {
          console.error(data.message);
          this.stopResponse();
        }
      }
      catch (e) {
        console.error("Failed to parse WS message", e);
      }
    };

    this.ws.onerror = (err) => {
      console.error("Transcription WS error", err);
      this.stopResponse();
    };

    this.ws.onclose = () => {
      this.isStreaming = false;
    };

    try {
      this.audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new AudioContext({ sampleRate: 16000 });
      const input = audioContext.createMediaStreamSource(this.audioStream);
      const processor = audioContext.createScriptProcessor(1024, 1, 1);

      input.connect(processor);
      processor.connect(audioContext.destination);

      processor.onaudioprocess = (e) => {
        if (!this.isStreaming || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;

        const float32Array = e.inputBuffer.getChannelData(0);
        const int16Array = new Int16Array(float32Array.length);

        for (let i = 0; i < float32Array.length; i++) {
          int16Array[i] = float32Array[i] < 0
            ? float32Array[i] * 0x8000
            : float32Array[i] * 0x7fff;
        }

        this.ws.send(int16Array.buffer);
      };
    } catch (err) {
      console.error(err);
      this.stopResponse();
    }
  }

  _updateTranscript(transcript, isPartial) {
    const textArea = this.messageInput?.textarea;
    if (!textArea) return;

    this.messageInput?.reset();

    if (isPartial) {
      textArea.appendChild(document.createTextNode(
        `${this.lastFinalTranscript} ${transcript}`
      ));
    } else {
      this.lastFinalTranscript += ` ${transcript}`.trim();
      textArea.appendChild(document.createTextNode(this.lastFinalTranscript));
    }
  }

  stopResponse() {
    this.stopRecording();
    if (this.isStreaming) {
      console.log("Stopping response stream");
      document.dispatchEvent(new CustomEvent("stop-streaming"));
      this.isStreaming = false;
      this.enableSubmit();
      if (this.ws) {
        this.ws.close();
        this.ws = null;
      }
    }
  }

  stopRecording() {
    if (!this.isRecording) return console.warn("No active recording to stop");

    this.showRecordButton();
    console.log("Stopping stream");
    this.isRecording = false;
    if (this.audioStream) {
      const tracks = this.audioStream.getTracks();
      tracks.forEach((track) => {
        console.log("Stopping track:", track);
        track.stop();
      });
      this.audioStream = null;
      if (this.ws) {
        this.ws.close();
        this.ws = null;
      }
    }
  }


  /**
   * Show Dictate button and hide stop dictate button
   */
  showRecordButton() {
    showElement(this.buttonRecord);
    hideElement(this.buttonRecordStop);
  }


  /**
   * Show Send button and hide stop send button
   */
  showSendButton() {
    showElement(this.buttonSend);
    hideElement(this.buttonSendStop);
  }


  /**
   * Hide Dictate button and show stop dictate button
   */
  hideRecordButton() {
    hideElement(this.buttonRecord);
    showElement(this.buttonRecordStop);
  }


  /**
   * Hide Send button and show stop send button
   */
  hideSendButton() {
    hideElement(this.buttonSend);
    showElement(this.buttonSendStop);
  }


  /**
   * Disables submission
   */
  disableSubmit = () => {
    this.submitDisabled = true;
    if (this.buttonSend) this.buttonSend.disabled = true;
    if (this.buttonRecord) this.buttonRecord.disabled = true;
  };


  /**
   * Enables submission
   */
  enableSubmit = () => {
    this.submitDisabled = false;
    if (this.buttonSend) this.buttonSend.disabled = false;
    if (this.buttonRecord) this.buttonRecord.disabled = false;
  };
}

customElements.define("send-message-with-dictation", SendMessageWithDictation);

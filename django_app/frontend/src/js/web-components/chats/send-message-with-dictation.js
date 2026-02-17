// @ts-check
import { TranscribeStreamingClient, StartStreamTranscriptionCommand } from "@aws-sdk/client-transcribe-streaming";
import { Readable } from 'readable-stream'
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

    customElements.whenDefined("rbds-message-input").then(() => this.#bindEvents())
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
    return /** @type {MessageInput} */ (document.querySelector('rbds-message-input'));
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

    const client = new TranscribeStreamingClient({
      region: "eu-west-2",
      credentials: async () => {
        const response = await (await fetch('/api/v0/aws-credentials', {
          method: 'GET',
          headers: {
              'X-API-KEY': this._apiKey || "",
              'Content-Type': 'application/json'
          }
        })).json();
        return {
          accessKeyId: response.AccessKeyId,
          secretAccessKey: response.SecretAccessKey,
          sessionToken: response.SessionToken || undefined,
        };
      },
    });

    try {
      this.audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new AudioContext({ sampleRate: 16000 });
      const input = audioContext.createMediaStreamSource(this.audioStream);
      const processor = audioContext.createScriptProcessor(1024, 1, 1);

      input.connect(processor);
      processor.connect(audioContext.destination);

      const readableStream = new Readable();
      processor.onaudioprocess = (e) => {
        const float32Array = e.inputBuffer.getChannelData(0);
        const int16Array = new Int16Array(float32Array.length);
        float32Array.forEach((val, idx) => {
          int16Array[idx] = val < 0 ? val * 0x8000 : val * 0x7fff;
        });
        readableStream.emit("audioData", new Int8Array(int16Array.buffer));
      };

      const audioStream = async function* () {
        while (this.isStreaming) {
          const chunk = await new Promise((resolve) =>
            readableStream.once("audioData", resolve)
          );
          yield { AudioEvent: { AudioChunk: chunk } };
        }
      }.bind(this);

      const command = new StartStreamTranscriptionCommand({
        LanguageCode: "en-GB",
        MediaSampleRateHertz: 16000,
        MediaEncoding: "pcm",
        AudioStream: audioStream(),
      });

      const response = await client.send(command);
      const textArea = this.messageInput?.textarea;

      let lastFinalTranscript = "";
      let partialTranscript = "";

      if (!response.TranscriptResultStream) throw new Error("No TranscriptResultStream returned from AWS");

      for await (const event of response.TranscriptResultStream) {
        if (event.TranscriptEvent) {
          const results = event.TranscriptEvent.Transcript?.Results;

          results?.forEach((result) => {
            const transcript = (result.Alternatives || [])
              .map((alt) => alt.Transcript)
              .join(" ");

            if (result.IsPartial) {
              partialTranscript = transcript;
              this.messageInput?.reset();
              textArea?.appendChild(document.createTextNode(`${lastFinalTranscript} ${partialTranscript}`));
            } else {
              lastFinalTranscript += ` ${transcript}`;
              partialTranscript = "";
              this.messageInput?.reset();
              textArea?.appendChild(document.createTextNode(lastFinalTranscript.trim()));
            }
          });
        }
      }
    } catch (error) {
      console.error("Error starting streaming:", error);
      this.isStreaming = false;
      this.showRecordButton();
      this.showSendButton();
      alert("Microphone usage is not permitted");
    }
  }

  stopResponse() {
    this.stopRecording();
    if (this.isStreaming) {
      console.log("Stopping response stream");
      document.dispatchEvent(new CustomEvent("stop-streaming"));
      this.isStreaming = false;
      this.enableSubmit();
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

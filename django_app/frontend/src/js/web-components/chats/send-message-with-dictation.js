// @ts-check
import { TranscribeStreamingClient, StartStreamTranscriptionCommand } from "@aws-sdk/client-transcribe-streaming";
import { Readable } from 'readable-stream'

class SendMessageWithDictation extends HTMLElement {

  connectedCallback() {
    const apiKey = this.getAttribute("data-api-key");
    this.apiKey = apiKey;
    this.removeAttribute("data-api-key");

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
  
    this.isStreaming = true;
    this.buttonSend.style.display = "none";
    this.buttonRecord.style.display = "none";
    this.buttonStop.style.display = "flex";
  
    const client = new TranscribeStreamingClient({
      region: "eu-west-2",
      credentials: async () => {
        const response = await (await fetch('/api/v0/aws-credentials', {
          method: 'GET',
          headers: {
              'X-API-KEY': this.apiKey,
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
      const textArea = document.querySelector("#message");
  
      let lastFinalTranscript = "";
      let partialTranscript = "";
  
      for await (const event of response.TranscriptResultStream) {
        if (event.TranscriptEvent) {
          const results = event.TranscriptEvent.Transcript.Results;
  
          results.forEach((result) => {
            const transcript = (result.Alternatives || [])
              .map((alt) => alt.Transcript)
              .join(" ");
  
            if (result.IsPartial) {
              partialTranscript = transcript;
              textArea.innerText = `${lastFinalTranscript} ${partialTranscript}`;
            } else {
              lastFinalTranscript += ` ${transcript}`;
              partialTranscript = "";
              textArea.innerText = lastFinalTranscript.trim();
            }
          });
        }
      }
    } catch (error) {
      console.error("Error starting streaming:", error);
      this.isStreaming = false;
      this.buttonRecord.style.display = "none";
      this.buttonSend.style.display = "flex";
      this.buttonStop.style.display = "none";
      alert("Microphone usage is not permitted");
    }
  }
  

  stopAction() {
    if (this.isRecording && this.mediaRecorder) {
      console.log("Stopped recording");
      this.mediaRecorder.stop();
      this.isRecording = false;
    } else if (this.isStreaming) {
      console.log("Stopping stream");
      document.dispatchEvent(new CustomEvent("stop-streaming"));
      this.isStreaming = false;
  
      if (this.audioStream) {
        const tracks = this.audioStream.getTracks();
        tracks.forEach((track) => {
          console.log("Stopping track:", track);
          track.stop();
        });
        this.audioStream = null;
      }
    } else {
      console.warn("No active recording or streaming to stop");
    }
  
    this.buttonStop.style.display = "none";
    this.buttonRecord.style.display = "none";
    this.buttonSend.style.display = "flex";
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

customElements.define("send-message-with-dictation", SendMessageWithDictation);
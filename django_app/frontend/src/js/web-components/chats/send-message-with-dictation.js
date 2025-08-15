// @ts-check
import { TranscribeStreamingClient, StartStreamTranscriptionCommand } from "@aws-sdk/client-transcribe-streaming";
import { Readable } from 'readable-stream'

class SendMessageWithDictation extends HTMLElement {
  connectedCallback() {
    const apiKey = this.getAttribute("data-api-key");
    this.apiKey = apiKey;
    this.removeAttribute("data-api-key");

    this.buttonRecord = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(1)")
    );
    this.buttonSend = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(3)")
    );
    this.buttonStop = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(2)")
    );
    this.buttonSendStop = /** @type {HTMLButtonElement} */ (
      this.querySelector("button:nth-child(4)")
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
    const textArea = document.querySelector("#message");
    textArea.innerHTML = ''

    this.isStreaming = true;
    this.isStreaming = true;
    this.buttonRecord.style.display = "none";
    this.buttonStop.style.display = "inline";

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
              textArea.innerHTML = `${lastFinalTranscript} ${partialTranscript}`;
            } else {
              lastFinalTranscript += ` ${transcript}`;
              partialTranscript = "";
              textArea.innerHTML = lastFinalTranscript.trim();
            }
          });
        }
      }
    } catch (error) {
      console.error("Error starting streaming:", error);
      this.isStreaming = false;
      this.buttonRecord.style.display = "inline";
      this.buttonSend.style.display = "inline";
      this.buttonStop.style.display = "none";
      this.buttonSendStop.style.display = "inline";
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
    } else if (this.isStreaming) {
      this.showRecordButton();
      this.buttonRecord.disabled = true
      console.log("Stopping stream");
      this.isStreaming = false
      document.dispatchEvent(new CustomEvent("stop-streaming"));
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
  }

  showRecordButton() {
    if (!this.buttonRecord || !this.buttonStop) return;

    this.buttonRecord.style.display = "inline";
    this.buttonStop.style.display = "none";
  }
}

customElements.define("send-message-with-dictation", SendMessageWithDictation);

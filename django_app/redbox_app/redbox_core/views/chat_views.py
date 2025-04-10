import logging
import json
import uuid
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from http import HTTPStatus
from itertools import groupby
from operator import attrgetter
from pathlib import Path

from dataclasses_json import Undefined, dataclass_json
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from vosk import KaldiRecognizer, Model
from yarl import URL

from redbox_app.redbox_core.models import Chat, ChatLLMBackend, ChatMessage, File

logger = logging.getLogger(__name__)


import subprocess

@method_decorator(csrf_exempt, name="dispatch")
class TranscribeAudioView(View):
    def post(self, request):
        try:
            if "audio" not in request.FILES:
                logger.error("No audio found")
                return JsonResponse({"error": "No audio found"}, status=400)

            audio_file = request.FILES["audio"]

            input_path = "input_audio.webm"
            output_path = "input_audio.wav"

            try:
                with Path(input_path).open("wb") as f:
                    for chunk in audio_file.chunks():
                        f.write(chunk)
            except Exception as e:
                logger.exception(f"failed to save audio file: {e}")
                return JsonResponse({"error": "failed to save audio file"}, status=500)

            try:
                logger.info("Converting audio to WAV format using ffmpeg")
                subprocess.run(
                    ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
                    check=True
                )
                logger.info("Audio conversion successful")
            except subprocess.CalledProcessError as e:
                logger.exception(f"Audio conversion failed: {e}")
                return JsonResponse({"error": "failed to convert audio file"}, status=500)

            try:
                model = Model("vosk-model-small-en-us-0.15")

                wf = wave.open(output_path, "rb")
                logger.info(f"channels={wf.getnchannels()}, sample width={wf.getsampwidth()}, framerate={wf.getframerate()}")

                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000]:
                    logger.error("Invalid audio format for Vosk")
                    return JsonResponse({"error": "Invalid audio format"}, status=400)

                logger.info("starting transcription")
                logger.debug(f"Converted WAV file size: {Path(output_path).stat().st_size} bytes")
                rec = KaldiRecognizer(model, wf.getframerate())

                transcription = ""
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        result = rec.Result()
                        logger.debug(f"intermediate result: {result}")
                        transcription += json.loads(result).get("text", "")
                    else:
                        logger.debug(f"partial result: {rec.PartialResult()}")

                wf.close()
                Path(output_path).unlink()
                logger.info(f"Transcription completed successfully {transcription}")
                return JsonResponse({"transcription": transcription}, status=200)

            except Exception as e:
                logger.exception(f"Failed during transcription: {e}")
                return JsonResponse({"error": "Dictation not picked up"}, status=500)

        except Exception as e:
            logger.exception(f"An exceptional error occurred: {e}")
            return None



class ChatsView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest, chat_id: uuid.UUID | None = None) -> HttpResponse:
        chat = Chat.get_ordered_by_last_message_date(request.user)

        messages: Sequence[ChatMessage] = []
        current_chat = None
        if chat_id:
            current_chat = get_object_or_404(Chat, id=chat_id)
            if current_chat.user != request.user:
                return redirect(reverse("chats"))
            messages = ChatMessage.get_messages_ordered_by_citation_priority(chat_id)

        endpoint = URL.build(
            scheme=settings.WEBSOCKET_SCHEME,
            host="localhost" if settings.ENVIRONMENT.is_test else settings.ENVIRONMENT.hosts[0],
            port=int(request.META["SERVER_PORT"]) if settings.ENVIRONMENT.is_test else None,
            path=r"/ws/chat/",
        )

        completed_files, processing_files = File.get_completed_and_processing_files(request.user)

        self.decorate_selected_files(completed_files, messages)
        chat_grouped_by_date_group = groupby(chat, attrgetter("date_group"))

        chat_backend = current_chat.chat_backend if current_chat else ChatLLMBackend.objects.get(is_default=True)

        # Add footnotes to messages
        for message in messages:
            footnote_counter = 1
            for display, href, text_in_answer in message.unique_citation_uris():  # noqa: B007
                if text_in_answer:
                    message.text = message.text.replace(
                        text_in_answer,
                        f'{text_in_answer}<a class="rb-footnote-link" href="#footnote-{message.id}-{footnote_counter}">{footnote_counter}</a>',  # noqa: E501
                    )
                    footnote_counter = footnote_counter + 1

        context = {
            "chat_id": chat_id,
            "messages": messages,
            "chat_grouped_by_date_group": chat_grouped_by_date_group,
            "current_chat": current_chat,
            "streaming": {"endpoint": str(endpoint)},
            "contact_email": settings.CONTACT_EMAIL,
            "completed_files": completed_files,
            "processing_files": processing_files,
            "chat_title_length": settings.CHAT_TITLE_LENGTH,
            "llm_options": [
                {
                    "name": str(chat_llm_backend),
                    "default": chat_llm_backend.is_default,
                    "selected": chat_llm_backend == chat_backend,
                    "id": chat_llm_backend.id,
                }
                for chat_llm_backend in ChatLLMBackend.objects.filter(enabled=True)
            ],
        }

        return render(
            request,
            template_name="chats.html",
            context=context,
        )

    @staticmethod
    def decorate_selected_files(all_files: Sequence[File], messages: Sequence[ChatMessage]) -> None:
        if messages:
            last_user_message = [m for m in messages if m.role == ChatMessage.Role.user][-1]
            selected_files: Sequence[File] = last_user_message.selected_files.all() or []
        else:
            selected_files = []

        for file in all_files:
            file.selected = file in selected_files


class ChatsTitleView(View):
    @dataclass_json(undefined=Undefined.EXCLUDE)
    @dataclass(frozen=True)
    class Title:
        name: str

    @method_decorator(login_required)
    def post(self, request: HttpRequest, chat_id: uuid.UUID) -> HttpResponse:
        chat: Chat = get_object_or_404(Chat, id=chat_id)
        user_rating = ChatsTitleView.Title.schema().loads(request.body)

        chat.name = user_rating.name
        chat.save(update_fields=["name"])

        return HttpResponse(status=HTTPStatus.NO_CONTENT)


class UpdateChatFeedback(View):
    @method_decorator(login_required)
    def post(self, request: HttpRequest, chat_id: uuid.UUID) -> HttpResponse:
        def convert_to_boolean(value: str):
            return value == "Yes"

        chat: Chat = get_object_or_404(Chat, id=chat_id)
        chat.feedback_achieved = convert_to_boolean(request.POST.get("achieved"))
        chat.feedback_saved_time = convert_to_boolean(request.POST.get("saved_time"))
        chat.feedback_improved_work = convert_to_boolean(request.POST.get("improved_work"))
        chat.feedback_notes = request.POST.get("notes")
        chat.save()
        return HttpResponse(status=HTTPStatus.NO_CONTENT)


class DeleteChat(View):
    @method_decorator(login_required)
    def post(self, request: HttpRequest, chat_id: uuid.UUID) -> HttpResponse:  # noqa: ARG002
        chat: Chat = get_object_or_404(Chat, id=chat_id)
        chat.archived = True
        chat.save()
        return HttpResponse(status=HTTPStatus.NO_CONTENT)
